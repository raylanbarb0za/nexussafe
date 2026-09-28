"""
NEXUS-SAFE: Protótipo de Plataforma de Segurança Cognitiva Para Identificação e Predição de Riscos em Ambientes Industriais
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

# ================================================================================
# 1. CLASSE DO GRAFO DE RISCO EMERGENTE
# ================================================================================

class GrafoRiscoEmergente:
    """Representa as correlações entre fatores de risco em um ambiente industrial"""
    
    def __init__(self):
        self.G = nx.DiGraph()
        self._criar_grafo()
    
    def _criar_grafo(self):
        """Cria o grafo com fatores de risco e suas correlações"""
        
        # Adicionar nós (fatores de risco)
        fatores = {
            'temp': 'Temperatura Elevada (>70°C)',
            'vibra': 'Vibração Alta (>7)',
            'pressao': 'Pressão Elevada (>20bar)',
            'fadiga': 'Fadiga do Operador (>7)',
            'umidade': 'Umidade Anômala',
            'risco_critico': '⚠️ RISCO CRÍTICO'
        }
        
        for chave, descricao in fatores.items():
            self.G.add_node(chave, label=descricao)
        
        # Adicionar arestas (pesos = força da correlação)
        # Formato: (origem, destino, peso_correlacao)
        correlacoes = [
            # Riscos individuais -> Risco crítico
            ('temp', 'risco_critico', 0.30),       # Temperatura sozinha = 30% risco
            ('vibra', 'risco_critico', 0.35),      # Vibração = 35%
            ('pressao', 'risco_critico', 0.25),    # Pressão = 25%
            ('fadiga', 'risco_critico', 0.28),     # Fadiga = 28%
            ('umidade', 'risco_critico', 0.15),    # Umidade = 15%
            
            # Correlações entre fatores (amplificam risco)
            ('temp', 'vibra', 0.60),                # Temp + Vibra = FORTE
            ('vibra', 'fadiga', 0.50),              # Vibra + Fadiga = FORTE
            ('pressao', 'temp', 0.45),              # Pressão + Temp = MÉDIA
            ('umidade', 'temp', 0.40),              # Umidade + Temp = MÉDIA
        ]
        
        for origem, destino, peso in correlacoes:
            self.G.add_edge(origem, destino, weight=peso)

    def calcular_risco_correlado(self, leitura_sensores):

        risco_total = 0

        # PASSO 1: Calcular risco individual por fator
        risco_temp = 0.30 if leitura_sensores.get('temperatura', 0) > 70 else 0
        risco_vibra = 0.35 if leitura_sensores.get('vibracao', 0) > 7 else 0
        risco_pressao = 0.25 if leitura_sensores.get('pressao', 0) > 20 else 0
        risco_fadiga = 0.28 if leitura_sensores.get('fadiga_operador', 0) > 7 else 0
        risco_umidade = 0.15 if leitura_sensores.get('umidade', 0) > 80 or leitura_sensores.get('umidade', 0) < 20 else 0

        # PASSO 2: Somar riscos individuais
        risco_total = risco_temp + risco_vibra + risco_pressao + risco_fadiga + risco_umidade

        # PASSO 3: Aplicar amplificações por correlações (RISCO EMERGENTE)
        # Quando múltiplos fatores ocorrem simultaneamente, o risco AMPLIFICA

        if leitura_sensores.get('temperatura', 0) > 70 and leitura_sensores.get('vibracao', 0) > 7:
            # Correlação Temp + Vibra = amplifica em 60%
            risco_total *= (1 + 0.60)

        if leitura_sensores.get('vibracao', 0) > 7 and leitura_sensores.get('fadiga_operador', 0) > 7:
            # Correlação Vibra + Fadiga = amplifica em 50%
            risco_total *= (1 + 0.50)

        if leitura_sensores.get('pressao', 0) > 20 and leitura_sensores.get('temperatura', 0) > 70:
            # Correlação Pressão + Temp = amplifica em 45%'
            risco_total *= (1 + 0.45)

        # Limitar entre 0 e 1
        return min(risco_total, 1.0)

    def obter_info_grafo(self):
        """Retorna informações sobre o grafo"""
        return {
            'total_nos': self.G.number_of_nodes(),
            'total_arestas': self.G.number_of_edges(),
            'nodes': dict(self.G.nodes(data=True)),
            'edges': [(u, v, d['weight']) for u, v, d in self.G.edges(data=True)]
        }


# ================================================================================
# 2. CLASSE DE INTELIGÊNCIA ARTIFICIAL (DETECÇÃO DE ANOMALIAS)
# ================================================================================

class DetectorAnomaliasIA:
    """
    Usa Isolation Forest para detectar padrões anômalos nos dados
    """
    
    def __init__(self, taxa_contaminacao=0.10):
        self.modelo = IsolationForest(contamination=taxa_contaminacao, random_state=42)
        self.scaler = StandardScaler()
        self.features_treinadas = None
        self.eh_treinado = False
    
    def treinar(self, dados):
        """
        Treina o modelo com dados históricos
        
        Args:
            dados: DataFrame com colunas de sensores
        """
        features = ['temperatura', 'vibracao', 'pressao', 'fadiga_operador', 'umidade']
        X = dados[features].values
        
        # Normalizar dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Treinar modelo
        self.modelo.fit(X_scaled)
        self.features_treinadas = features
        self.eh_treinado = True
        
        print(f" Modelo IA treinado com {len(dados)} amostras")
    
    def detectar_anomalias(self, dados):
        """
        Detecta anomalias em novos dados
        
        Args:
            dados: DataFrame com novos sensores
        
        Returns:
            array: -1 para anomalia, 1 para normal
        """
        if not self.eh_treinado:
            raise ValueError("Modelo não foi treinado! Execute treinar() primeiro.")
        
        features = self.features_treinadas
        X = dados[features].values
        X_scaled = self.scaler.transform(X)
        
        predicoes = self.modelo.predict(X_scaled)
        return predicoes == -1  # True se anomalia
    
    def calcular_score_anomalia(self, dados):
        """
        Calcula score de anomalia (0 a 1) para cada amostra
        
        Returns:
            array: scores entre 0 e 1
        """
        if not self.eh_treinado:
            raise ValueError("Modelo não foi treinado!")
        
        features = self.features_treinadas
        X = dados[features].values
        X_scaled = self.scaler.transform(X)
        
        # Score da anomalia (quanto mais negativo, mais anômalo)
        scores = -self.modelo.score_samples(X_scaled)
        # Normalizar entre 0 e 1
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores


# ================================================================================
# 3. CLASSE PRINCIPAL DO NEXUS-SAFE
# ================================================================================

class NEXUS_SAFE:
    """
    Sistema completo de detecção e predição de riscos
    """
    
    def __init__(self):
        self.grafo = GrafoRiscoEmergente()
        self.detector_ia = DetectorAnomaliasIA(taxa_contaminacao=0.08)
        self.historico = []
        self.alertas = []
    
    def gerar_dados_simulados(self, n_leituras=200, seed=42):
        """
        Gera dados simulados de sensores industriais
        
        Args:
            n_leituras: número de leituras a gerar
            seed: seed para reprodutibilidade
        
        Returns:
            DataFrame com dados simulados
        """
        np.random.seed(seed)
        
        # Valores normais (com pequenas variações)
        dados = {
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_leituras)],
            'temperatura': np.random.normal(45, 3, n_leituras),      # Média 45°C
            'vibracao': np.random.normal(2.2, 0.4, n_leituras),      # Média 2.2
            'pressao': np.random.normal(10.5, 0.8, n_leituras),      # Média 10.5 bar
            'fadiga_operador': np.random.normal(3, 1, n_leituras),   # Média 3
            'umidade': np.random.normal(50, 8, n_leituras),          # Média 50%
        }
        
        df = pd.DataFrame(dados)
        
        # Adicionar ANOMALIAS REALISTAS (situações de risco)
        indices_anomalias = np.random.choice(n_leituras, size=int(n_leituras*0.08), replace=False)
        
        for idx in indices_anomalias:
            tipo_anomalia = np.random.choice(['temp_alta', 'vibra_alta', 'correlacao'])
            
            if tipo_anomalia == 'temp_alta':
                df.loc[idx, 'temperatura'] = np.random.uniform(72, 85)
            elif tipo_anomalia == 'vibra_alta':
                df.loc[idx, 'vibracao'] = np.random.uniform(7.5, 9)
            else:  # correlacao (múltiplos fatores)
                df.loc[idx, 'temperatura'] = np.random.uniform(70, 80)
                df.loc[idx, 'vibracao'] = np.random.uniform(7, 9)
                df.loc[idx, 'fadiga_operador'] = np.random.uniform(7, 10)
        
        # Manter valores positivos (apenas colunas numéricas)
        colunas_numericas = ['temperatura', 'vibracao', 'pressao', 'fadiga_operador', 'umidade']
        df[colunas_numericas] = df[colunas_numericas].clip(lower=0)
        
        return df
    
    def processar_dados(self, dados):
        """
        Pipeline completo de análise dos dados
        
        Args:
            dados: DataFrame com leituras de sensores
        
        Returns:
            DataFrame com análises completas
        """
        # ETAPA 1: Treinar modelo IA
        print("\n Etapa 1: Treinando modelo de IA...")
        self.detector_ia.treinar(dados)
        
        # ETAPA 2: Detectar anomalias pela IA
        print("Etapa 2: Detectando anomalias...")
        dados['anomalia_ia'] = self.detector_ia.detectar_anomalias(dados)
        dados['score_anomalia'] = self.detector_ia.calcular_score_anomalia(dados)
        
        # ETAPA 3: Calcular risco correlado pelo grafo
        print("Etapa 3: Calculando risco correlado...")
        dados['risco_correlado'] = dados.apply(
            lambda row: self.grafo.calcular_risco_correlado(row.to_dict()),
            axis=1
        )
        
        # ETAPA 4: Classificar nível de alerta
        print("⚠️Etapa 4: Classificando alertas...")
        
        def classificar_alerta(row):
            if row['risco_correlado'] > 0.70 or row['anomalia_ia']:
                return 'CRÍTICO'
            elif row['risco_correlado'] > 0.40:
                return 'ALERTA'
            elif row['anomalia_ia']:
                return 'ATENÇÃO'
            else:
                return 'NORMAL'
        
        dados['nivel_alerta'] = dados.apply(classificar_alerta, axis=1)
        
        # ETAPA 5: Identificar fatores de risco presentes
        print("Etapa 5: Identificando fatores de risco...")
        
        def identificar_fatores(row):
            fatores = []
            if row['temperatura'] > 70:
                fatores.append('Temperatura Elevada')
            if row['vibracao'] > 7:
                fatores.append('Vibração Alta')
            if row['pressao'] > 20:
                fatores.append('Pressão Elevada')
            if row['fadiga_operador'] > 7:
                fatores.append('Fadiga do Operador')
            if row['umidade'] > 80 or row['umidade'] < 20:
                fatores.append('Umidade Anômala')
            return ', '.join(fatores) if fatores else 'Nenhum'
        
        dados['fatores_presentes'] = dados.apply(identificar_fatores, axis=1)
        
        self.historico = dados
        return dados
    
    def gerar_relatorio_resumido(self):
        """Gera relatório resumido dos resultados"""
        if self.historico.empty:
            print("Nenhum dado processado!")
            return
        
        print("\n" + "="*70)
        print("RELATÓRIO EXECUTIVO - NEXUS-SAFE")
        print("="*70)
        
        print(f"\nESTATÍSTICAS GERAIS:")
        print(f"  • Total de leituras analisadas: {len(self.historico)}")
        print(f"  • Período: {self.historico['timestamp'].min()} a {self.historico['timestamp'].max()}")
        
        print(f"\nDETECÇÃO DE ANOMALIAS (IA):")
        anomalias_detectadas = self.historico['anomalia_ia'].sum()
        print(f"  • Anomalias detectadas: {anomalias_detectadas} ({anomalias_detectadas/len(self.historico)*100:.1f}%)")
        
        print(f"\nDISTRIBUIÇÃO DE ALERTAS:")
        distribuicao = self.historico['nivel_alerta'].value_counts()
        for alerta, qtd in distribuicao.items():
            percentual = qtd / len(self.historico) * 100
            simbolos = {'CRÍTICO': '!!!', 'ALERTA': '!', 'ATENÇÃO': '?', 'NORMAL': 'OK'}
            print(f" {simbolos.get(alerta, '•'):6} {alerta:12} : {qtd:3} ({percentual:5.1f}%)")
        
        print(f"\nANÁLISE DE RISCO:")
        print(f"  • Risco médio: {self.historico['risco_correlado'].mean()*100:6.2f}%")
        print(f"  • Risco mínimo: {self.historico['risco_correlado'].min()*100:6.2f}%")
        print(f"  • Risco máximo: {self.historico['risco_correlado'].max()*100:6.2f}%")
        
        print(f"\nFATORES DE RISCO MAIS FREQUENTES:")
        todos_fatores = []
        for fatores_str in self.historico['fatores_presentes']:
            if fatores_str != 'Nenhum':
                todos_fatores.extend(fatores_str.split(', '))
        
        if todos_fatores:
            fator_counts = pd.Series(todos_fatores).value_counts()
            for fator, qtd in fator_counts.head(5).items():
                print(f"  • {fator:25}: {qtd:3} ocorrências")
        
        print("\n" + "="*70)
    
    def gerar_relatorio_detalhado(self, n_casos=10):
        """
        Gera relatório com casos específicos de risco
        
        Args:
            n_casos: número de casos críticos a mostrar
        """
        if self.historico.empty:
            print("Nenhum dado processado!")
            return
        
        print("\n" + "="*70)
        print("🔍 CASOS CRÍTICOS IDENTIFICADOS")
        print("="*70)
        
        # Filtrar casos críticos
        criticos = self.historico[self.historico['nivel_alerta'] == 'CRÍTICO'].head(n_casos)
        
        if len(criticos) == 0:
            print("✓ Nenhum caso crítico encontrado!")
            return
        
        for idx, (i, row) in enumerate(criticos.iterrows(), 1):
            print(f"\nCASO #{idx}")
            print(f"  Timestamp: {row['timestamp']}")
            print(f"  Nível de Alerta: {row['nivel_alerta']}")
            print(f"  Risco Correlado: {row['risco_correlado']*100:.1f}%")
            print(f"  Score Anomalia (IA): {row['score_anomalia']:.3f}")
            print(f"  Fatores Presentes: {row['fatores_presentes']}")
            print(f"  \n  Leituras dos Sensores:")
            print(f"    • Temperatura: {row['temperatura']:.1f}°C")
            print(f"    • Vibração: {row['vibracao']:.2f}")
            print(f"    • Pressão: {row['pressao']:.2f} bar")
            print(f"    • Fadiga Operador: {row['fadiga_operador']:.1f}")
            print(f"    • Umidade: {row['umidade']:.1f}%")
        
        print("\n" + "="*70)
    
    def gerar_grafico_timeline(self):
        """Gera visualização de risco ao longo do tempo"""
        if self.historico.empty:
            return
        
        # Usar os últimos 50 registros para melhor visualização
        dados_plot = self.historico.tail(50).copy()
        dados_plot['index'] = range(len(dados_plot))
        
        print("\n" + "="*70)
        print("📈 TIMELINE DE RISCO (Últimas 50 leituras)")
        print("="*70)
        
        # Escala simplificada
        for idx, row in dados_plot.iterrows():
            risco_pct = row['risco_correlado']
            num_barras = int(risco_pct * 50)
            
            if risco_pct > 0.70:
                cor = "🔴"
            elif risco_pct > 0.40:
                cor = "🟠"
            elif risco_pct > 0.20:
                cor = "🟡"
            else:
                cor = "🟢"
            
            barra = "█" * num_barras + "░" * (50 - num_barras)
            print(f"{row['index']:3} {cor} {barra} {risco_pct*100:5.1f}%")
        
        print("="*70)


# ================================================================================
# 4. DEMONSTRAÇÃO E TESTE
# ================================================================================

def main():
    """Função principal - executa o protótipo completo"""
    print("\n" + "="*70)
    print("NEXUS-SAFE - PROTÓTIPO DE SISTEMA DE DETECÇÃO DE RISCOS")
    print("="*70)
    
    # Criar instância do sistema
    sistema = NEXUS_SAFE()
    
    # PASSO 1: Gerar dados simulados
    print("\nPASSO 1: Gerando dados simulados de sensores...")
    dados = sistema.gerar_dados_simulados(n_leituras=200)
    print(f"✓ {len(dados)} leituras geradas")
    print(f"  Período: {dados['timestamp'].min()} a {dados['timestamp'].max()}")
    
    # PASSO 2: Processar dados
    print("\nPASSO 2: Processando dados com Grafo + IA...")
    dados_analisados = sistema.processar_dados(dados)
    
    # PASSO 3: Gerar relatórios
    print("\nPASSO 3: Gerando relatórios...")
    sistema.gerar_relatorio_resumido()
    sistema.gerar_relatorio_detalhado(n_casos=5)
    sistema.gerar_grafico_timeline()
    
    # PASSO 4: Mostrar algumas linhas dos dados
    print("\n" + "="*70)
    print("AMOSTRA DOS DADOS PROCESSADOS (Primeiras 10 linhas)")
    print("="*70)
    
    colunas_principais = ['temperatura', 'vibracao', 'pressao', 'fadiga_operador', 
                          'risco_correlado', 'nivel_alerta', 'anomalia_ia']
    
    print(dados_analisados[colunas_principais].head(10).to_string())
    
    # PASSO 5: Exportar dados para arquivo
    print("\n" + "="*70)
    print("PASSO 4: Exportando resultados...")
    
    # Salvar CSV
    arquivo_csv = Path("nexus_safe_resultados.csv")
    dados_analisados.to_csv(arquivo_csv, index=False)
    print(f"✓ Dados exportados para: {arquivo_csv}")
    
    # Salvar JSON com resumo
    resumo = {
        'total_leituras': len(dados_analisados),
        'anomalias_detectadas': int(dados_analisados['anomalia_ia'].sum()),
        'risco_medio_pct': float(dados_analisados['risco_correlado'].mean() * 100),
        'risco_maximo_pct': float(dados_analisados['risco_correlado'].max() * 100),
        'distribuicao_alertas': dados_analisados['nivel_alerta'].value_counts().to_dict(),
    }
    
    arquivo_json = Path("nexus_safe_resumo.json")
    with open(arquivo_json, 'w', encoding='utf-8') as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(f"✓ Resumo exportado para: {arquivo_json}")
    
    print("\n" + "="*70)
    print("PROTÓTIPO EXECUTADO COM SUCESSO!")
    print("="*70)
    
    return sistema, dados_analisados


if __name__ == "__main__":
    sistema, dados = main()
