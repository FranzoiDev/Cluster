# Trabalho de Clusterização K-means e Preparação para KNN

## Análise de Padrões de Alta do Bitcoin

Este projeto implementa um sistema completo de clusterização utilizando a técnica K-means e preparação para aplicação da técnica KNN para identificar padrões de alta do Bitcoin.

## Objetivo

Implementar um processo de clusterização que explore cada uma das etapas envolvidas e prepare a base de dados para uma futura implementação do algoritmo KNN, especificamente para análise de dados de Bitcoin com as seguintes características:
- Dia do ano
- Ano
- Valor de fechamento do dia
- Volume em dólar do Bitcoin no dia
- Dados categóricos

## Estrutura do Projeto

### Arquivos Principais

- `cluster_analysis.py` - Implementação principal do sistema de clusterização
- `database_models.py` - Modelos de banco de dados para persistência
- `README.md` - Documentação do projeto

### Classes Principais

#### `Element`
Representa um elemento individual com:
- Features numéricas
- Dados categóricos opcionais
- Indicação se é centróide

#### `Cluster`
Representa um cluster com:
- Elementos pertencentes ao cluster
- Centróide virtual calculado
- Métodos para cálculo de dispersão
- Identificação de elementos distantes

#### `ClusterManager`
Gerencia a coleção de clusters com:
- Criação de clusters iniciais
- Atribuição de novos elementos
- Reorganização automática baseada em dispersão
- Conversão de dados categóricos

## Implementação dos 5 Tópicos da Atividade

### Tópico 1: Estrutura de dados inicial e manipulação
- **Inserção**: Método `add_new_record_to_system()`
- **Remoção**: Método `remove_record()`
- **Alteração**: Método `alter_record_features()`
- **Indicação de centróide**: Atributo `is_centroid` em cada elemento
- **Clusters iniciais**: Cada cluster inicia com um único elemento

### Tópico 2: Atribuição de elementos e cálculo de distâncias
- **Verificação de proximidade**: Cálculo de distância euclidiana para todos os clusters
- **Atribuição automática**: Elemento é adicionado ao cluster mais próximo
- **Atualização de centróide**: Recálculo automático após cada adição

### Tópico 3: Recálculo do centróide e reorganização
- **Recálculo automático**: Método `_recalculate_virtual_centroid()`
- **Atualização de marcação**: Método `_designate_element_as_centroid()`
- **Reorganização**: Método `_update_centroids()` consolida as operações

### Tópico 4: Análise de dispersão e criação de novos clusters
- **Limiar de dispersão**: Configurável via `dispersion_threshold`
- **Identificação de elementos distantes**: Método `get_distant_elements()`
- **Criação automática de novos clusters**: Quando elementos estão muito distantes
- **Reorganização completa**: Método `_check_and_reorganize_clusters()`

### Tópico 5: Adequação de dados para uso futuro com KNN
- **Suporte a dados categóricos**: Atributo `categorical_data` em elementos
- **Conversão automática**: Método `_process_categorical_data()` converte strings em valores numéricos
- **Normalização para KNN**: Função `normalize_features_for_knn()`
- **Implementação KNN completa**: Funções `find_k_nearest_neighbors()` e `predict_class_with_knn()`

## Como Usar

### Execução Básica
```bash
python3 cluster_analysis.py
```

### Exemplo de Uso Programático
```python
from cluster_analysis import ClusterManager

# Cria manager com limiar de dispersão
manager = ClusterManager(dispersion_threshold=500000000.0)

# Dados iniciais: [dia_do_ano, ano, valor_fechamento, volume_usd]
initial_data = [
    [15.0, 2023.0, 45000.0, 2000000000.0],
    [180.0, 2023.0, 65000.0, 4000000000.0]
]

# Cria clusters iniciais
manager.create_initial_clusters(initial_data)

# Adiciona novo elemento com dados categóricos
features = [30.0, 2023.0, 47000.0, 2200000000.0]
categorical = {"tendencia": "alta", "volatilidade": "baixa"}
element_id, cluster_id = manager.add_new_record_to_system(features, categorical)

# Visualiza detalhes dos clusters
details = manager.get_all_cluster_details()
print(details)
```

## Funcionalidades Especiais

### Conversão de Dados Categóricos
O sistema automaticamente converte dados categóricos em valores numéricos:
- `tendencia`: {"alta": 0.0, "correcao": 1.0}
- `volatilidade`: {"baixa": 0.0, "media": 1.0, "alta": 2.0}

### Reorganização Automática
Quando a dispersão de um cluster excede o limiar definido, o sistema:
1. Identifica elementos distantes do centróide
2. Remove esses elementos do cluster original
3. Cria um novo cluster com os elementos distantes
4. Recalcula todos os centróides

### KNN Integrado
O sistema inclui implementação completa de KNN com:
- Normalização de features (min-max)
- Busca dos k vizinhos mais próximos
- Predição por votação majoritária
- Suporte a dados categóricos

## Exemplo de Saída

```
=== EXEMPLO: Análise de Padrões de Alta do Bitcoin ===
Inicializados 2 clusters.
Clusters iniciais criados para análise de Bitcoin:
  {'id': 'cluster_1', 'virtual_centroid_features': [15.0, 2023.0, 45000.0, 2000000000.0], ...}
  {'id': 'cluster_2', 'virtual_centroid_features': [180.0, 2023.0, 65000.0, 4000000000.0], ...}

Adicionando novos dados de Bitcoin:
  Elemento abc123 -> Cluster cluster_1
  Elemento def456 -> Cluster cluster_1
  ...

Estado final dos clusters:
  Cluster cluster_1: 3 elementos, dispersão: 150000000.50
  Cluster cluster_2: 2 elementos, dispersão: 200000000.25

=== Demonstração KNN ===
Vizinhos mais próximos para [60.0, 2023.0, 50000.0, 2600000000.0]:
  1. Elemento xyz789 (distância: 0.1234) - Cluster: cluster_1
  2. Elemento uvw012 (distância: 0.2345) - Cluster: cluster_1
  3. Elemento rst345 (distância: 0.3456) - Cluster: cluster_2

Predição de cluster para novo ponto: cluster_1

Mapeamentos categóricos criados:
  tendencia: {'alta': 0.0, 'correcao': 1.0}
  volatilidade: {'baixa': 0.0, 'media': 1.0, 'alta': 2.0}
```

## Configurações Importantes

### Limiar de Dispersão
Para dados de Bitcoin, recomenda-se usar um limiar alto devido à magnitude dos valores:
```python
manager = ClusterManager(dispersion_threshold=500000000.0)
```

### Dimensões dos Dados
O sistema suporta expansão automática de dimensões quando dados categóricos são adicionados:
- Dados originais: 4 dimensões [dia, ano, preço, volume]
- Com categóricos: 6 dimensões [dia, ano, preço, volume, tendencia_num, volatilidade_num]

## Requisitos

- Python 3.7+

## Integrantes
- Gabriel Franzoi Barbosa de Souza, RA: 24463299-2
- Renata Heloisa Saraiva Casoni, R.A.: 22272753-2