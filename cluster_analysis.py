import math
import uuid
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter

class Logger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def log(self, message: str, level: str = "INFO"):
        if self.enabled:
            print(f"[{level}] {message}")

# Global logger instance
logger = Logger()

class Element:
    def __init__(self, features: List[float], element_id: Optional[str] = None, categorical_data: Optional[Dict[str, str]] = None):
        if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
            raise ValueError("Features devem ser uma lista de numeros.")
        if not features:
            raise ValueError("A lista de features nao pode estar vazia.")

        self.id: str = element_id if element_id else str(uuid.uuid4())
        self.features: List[float] = features
        self.is_centroid: bool = False
        self.categorical_data: Dict[str, str] = categorical_data or {}

    def __repr__(self) -> str:
        return f"Element(id={self.id}, features={self.features}, is_centroid={self.is_centroid})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "features": list(self.features),
            "is_centroid": self.is_centroid,
            "categorical_data": self.categorical_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Element':
        el = cls(
            features=data["features"],
            element_id=data["id"],
            categorical_data=data.get("categorical_data", {})
        )
        el.is_centroid = data["is_centroid"]
        return el

class Cluster:
    def __init__(self, cluster_id: str, initial_element_features: List[float]):
        if not initial_element_features:
            raise ValueError("Features do elemento inicial nao podem estar vazias")

        self.id: str = cluster_id
        self.elements: List[Element] = []
        self.virtual_centroid_features: List[float] = list(initial_element_features)

        # Adiciona o elemento inicial que forma o cluster
        initial_element = Element(features=initial_element_features)
        self._add_element_internal(initial_element)
        self._update_centroids()

    def _add_element_internal(self, element: Element):
        if not isinstance(element, Element):
            raise ValueError("Apenas objetos Element podem ser adicionados")
        self.elements.append(element)

    def _recalculate_virtual_centroid(self):
        # Recalcula o centroide virtual do cluster
        if not self.elements:
            self.virtual_centroid_features = [] # Indef se vazio
            return

        num_elements = len(self.elements)
        num_features = len(self.elements[0].features)

        sum_features = [0.0] * num_features
        for element in self.elements:
            for i in range(num_features):
                sum_features[i] += element.features[i]

        self.virtual_centroid_features = [s / num_elements for s in sum_features]

    def _designate_element_as_centroid(self):
        for el in self.elements:
            el.is_centroid = False

        if not self.elements or not self.virtual_centroid_features:
            return

        closest_element: Optional[Element] = None
        min_distance = float('inf')

        for element in self.elements:
            distance = self.euclidean_distance(element.features, self.virtual_centroid_features)
            if distance < min_distance:
                min_distance = distance
                closest_element = element

        if closest_element:
            closest_element.is_centroid = True

    def _update_centroids(self):
        # Auxiliar para consolidar os passos de atualizacao do centroide
        self._recalculate_virtual_centroid()
        self._designate_element_as_centroid()

    def add_element(self, element_features: List[float], categorical_data: Optional[Dict[str, str]] = None) -> Element:
        # Permite expansao das dimensoes do cluster para acomodar dados categoricos
        if self.virtual_centroid_features and len(element_features) > len(self.virtual_centroid_features):
            diff = len(element_features) - len(self.virtual_centroid_features)
            self.virtual_centroid_features.extend([0.0] * diff)

            for existing_element in self.elements:
                if len(existing_element.features) < len(element_features):
                    existing_diff = len(element_features) - len(existing_element.features)
                    existing_element.features.extend([0.0] * existing_diff)

        new_element = Element(features=element_features, categorical_data=categorical_data)
        self._add_element_internal(new_element)
        self._update_centroids()
        return new_element

    def remove_element(self, element_id: str) -> bool:
        element_to_remove = next((el for el in self.elements if el.id == element_id), None)
        if element_to_remove:
            self.elements.remove(element_to_remove)
            self._update_centroids()
            return True
        return False

    def update_element_features(self, element_id: str, new_features: List[float]) -> bool:
        found_element = next((el for el in self.elements if el.id == element_id), None)
        if found_element:
            if len(found_element.features) != len(new_features):
                raise ValueError("Dimensoes incompativeis")
            found_element.features = new_features
            self._update_centroids()
            return True
        return False

    def get_elements_data(self) -> List[Dict[str, Any]]:
        return [el.to_dict() for el in self.elements]

    def get_virtual_centroid_features(self) -> List[float]:
        return list(self.virtual_centroid_features) if self.virtual_centroid_features else []

    def calculate_dispersion(self) -> float:
        if not self.elements or not self.virtual_centroid_features:
            return 0.0

        total_distance = 0.0
        for element in self.elements:
            distance = self.euclidean_distance(element.features, self.virtual_centroid_features)
            total_distance += distance

        return total_distance / len(self.elements)

    def get_distant_elements(self, threshold: float) -> List[Element]:
        distant_elements = []
        for element in self.elements:
            distance = self.euclidean_distance(element.features, self.virtual_centroid_features)
            if distance > threshold:
                distant_elements.append(element)
        return distant_elements

    @staticmethod
    def euclidean_distance(features1: List[float], features2: List[float]) -> float:
        if not features1 or not features2:
            return float('inf')
        if len(features1) != len(features2):
            raise ValueError("Dimensoes incompativeis para calculo de distancia.")
        return math.sqrt(sum((f1 - f2)**2 for f1, f2 in zip(features1, features2)))

    def __repr__(self) -> str:
        designated_centroid_id = next((el.id for el in self.elements if el.is_centroid), "Nenhum")
        return (f"Cluster(id={self.id}, num_elements={len(self.elements)}, "
                f"virtual_centroid={self.virtual_centroid_features}, "
                f"designated_centroid_id={designated_centroid_id})")

class ClusterManager:
    def __init__(self, dispersion_threshold: float = 10.0):
        self.clusters: Dict[str, Cluster] = {}
        self.all_elements_map: Dict[str, Dict[str, Any]] = {}
        self._next_cluster_id_counter: int = 0
        self.dispersion_threshold = dispersion_threshold
        self.categorical_mappings: Dict[str, Dict[str, float]] = {}

    def _generate_cluster_id(self) -> str:
        self._next_cluster_id_counter += 1
        return f"cluster_{self._next_cluster_id_counter}"

    def create_initial_clusters(self, initial_element_features_list: List[List[float]]):
        # Cria clusters iniciais

        if self.clusters:
            raise Exception("Clusters iniciais já foram criados.")
        if not initial_element_features_list:
            raise ValueError("É necessário fornecer features para pelo menos um elemento inicial.")

        # Verificacao basica de dimen
        if len(initial_element_features_list) > 1:
            first_dim = len(initial_element_features_list[0])
            if not all(len(f) == first_dim for f in initial_element_features_list):
                raise ValueError("Todas as features devem ter a mesma dimensão.")

        for features in initial_element_features_list:
            cluster_id = self._generate_cluster_id()
            cluster = Cluster(cluster_id=cluster_id, initial_element_features=features)
            self.clusters[cluster_id] = cluster

            # O elemento inicial é criado dentro do construtor do cluster
            initial_element = cluster.elements[0]
            self.all_elements_map[initial_element.id] = {
                'element_obj': initial_element,
                'cluster_id': cluster_id
            }

        logger.log(f"Inicializados {len(self.clusters)} clusters.")

    def add_new_record_to_system(self, new_element_features: List[float],
                                categorical_data: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
        if not self.clusters:
            raise Exception("Nenhum cluster existe. Crie clusters iniciais primeiro")

        processed_features = self._process_categorical_data(new_element_features, categorical_data)

        # Encontra o cluster mais prox
        closest_cluster_id = self._find_closest_cluster(processed_features)

        if closest_cluster_id:
            target_cluster = self.clusters[closest_cluster_id]
            new_element_obj = target_cluster.add_element(processed_features, categorical_data)
            self.all_elements_map[new_element_obj.id] = {
                'element_obj': new_element_obj,
                'cluster_id': target_cluster.id
            }

            self._check_and_reorganize_clusters()

            logger.log(f"Elemento {new_element_obj.id} adicionado ao cluster {target_cluster.id}.")
            return new_element_obj.id, target_cluster.id
        else:
            raise Exception("Nao foi possivel encontrar o cluster mais proximo")

    def _find_closest_cluster(self, element_features: List[float]) -> Optional[str]:
        closest_cluster_id: Optional[str] = None
        min_distance = float('inf')

        for cid, cluster_obj in self.clusters.items():
            centroid_features = cluster_obj.get_virtual_centroid_features()

            # Ajusta as dimensoes se necessario
            if len(element_features) > len(centroid_features):
                comparison_features = element_features[:len(centroid_features)]
            else:
                comparison_features = element_features

            distance = Cluster.euclidean_distance(
                comparison_features,
                centroid_features
            )
            if distance < min_distance:
                min_distance = distance
                closest_cluster_id = cid

        return closest_cluster_id

    def _check_and_reorganize_clusters(self):
        clusters_to_reorganize = []

        for cluster_id, cluster in self.clusters.items():
            dispersion = cluster.calculate_dispersion()
            if dispersion > self.dispersion_threshold:
                distant_elements = cluster.get_distant_elements(self.dispersion_threshold)
                if len(distant_elements) >= 2:
                    clusters_to_reorganize.append((cluster_id, distant_elements))

        # Reorganiza clusters com alta dispersao
        for cluster_id, distant_elements in clusters_to_reorganize:
            self._create_new_cluster_from_distant_elements(cluster_id, distant_elements)

    def _create_new_cluster_from_distant_elements(self, original_cluster_id: str, distant_elements: List[Element]):
        if not distant_elements:
            return

        original_cluster = self.clusters[original_cluster_id]
        for element in distant_elements:
            original_cluster.remove_element(element.id)
            del self.all_elements_map[element.id]

        new_cluster_id = self._generate_cluster_id()
        first_element = distant_elements[0]
        new_cluster = Cluster(cluster_id=new_cluster_id, initial_element_features=first_element.features)

        new_cluster.elements.clear()

        # Adiciona todos os elementos distantes ao novo cluster
        for element in distant_elements:
            new_cluster._add_element_internal(element)
            self.all_elements_map[element.id] = {
                'element_obj': element,
                'cluster_id': new_cluster_id
            }

        new_cluster._update_centroids()
        self.clusters[new_cluster_id] = new_cluster

        logger.log(f"Novo cluster {new_cluster_id} criado com {len(distant_elements)} elementos distantes do cluster {original_cluster_id}.")

    def _process_categorical_data(self, features: List[float],
                                 categorical_data: Optional[Dict[str, str]] = None) -> List[float]:
        if not categorical_data:
            return features

        processed_features = list(features)

        for category, value in categorical_data.items():
            if category not in self.categorical_mappings:
                self.categorical_mappings[category] = {}

            if value not in self.categorical_mappings[category]:
                next_value = len(self.categorical_mappings[category])
                self.categorical_mappings[category][value] = float(next_value)

            processed_features.append(self.categorical_mappings[category][value])

        return processed_features

    def remove_record(self, element_id: str) -> bool:
        if element_id not in self.all_elements_map:
            logger.log(f"Elemento {element_id} nao encontrado.")
            return False

        record_info = self.all_elements_map[element_id]
        cluster_id = record_info['cluster_id']
        cluster = self.clusters.get(cluster_id)

        if cluster and cluster.remove_element(element_id):
            del self.all_elements_map[element_id]
            logger.log(f"Elemento {element_id} removido do cluster {cluster_id}.")
            return True
        return False

    def alter_record_features(self, element_id: str, new_features: List[float]) -> bool:
        if element_id not in self.all_elements_map:
            logger.log(f"Elemento {element_id} não encontrado.")
            return False

        record_info = self.all_elements_map[element_id]
        cluster_id = record_info['cluster_id']
        cluster = self.clusters.get(cluster_id)

        if cluster and cluster.update_element_features(element_id, new_features):
            logger.log(f"Features do elemento {element_id} atualizadas.")
            return True
        return False

    def get_cluster_details(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        cluster = self.clusters.get(cluster_id)
        if cluster:
            return {
                "id": cluster.id,
                "virtual_centroid_features": cluster.get_virtual_centroid_features(),
                "elements": cluster.get_elements_data(),
                "designated_centroid_id": next((el.id for el in cluster.elements if el.is_centroid), None),
                "dispersion": cluster.calculate_dispersion()
            }
        return None

    def get_all_cluster_details(self) -> Dict[str, Optional[Dict[str, Any]]]:
        return {cid: self.get_cluster_details(cid) for cid in self.clusters}

    def get_element_details(self, element_id: str) -> Optional[Dict[str, Any]]:
        if element_id in self.all_elements_map:
            record_info = self.all_elements_map[element_id]
            element_obj: Element = record_info['element_obj']
            return {
                "element_data": element_obj.to_dict(),
                "cluster_id": record_info['cluster_id']
            }
        return None

def normalize_features_for_knn(elements_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not elements_map:
        return elements_map

    # Coleta todas as features para normalizaçao
    all_features = []
    for data in elements_map.values():
        element_obj = data['element_obj']
        all_features.append(element_obj.features)

    if not all_features:
        return elements_map

    # Calcula min e max para cada dimensao
    num_features = len(all_features[0])
    min_vals = [float('inf')] * num_features
    max_vals = [float('-inf')] * num_features

    for features in all_features:
        for i, val in enumerate(features):
            min_vals[i] = min(min_vals[i], val)
            max_vals[i] = max(max_vals[i], val)

    normalized_map = {}
    for el_id, data in elements_map.items():
        element_obj = data['element_obj']
        normalized_features = []

        for i, val in enumerate(element_obj.features):
            if max_vals[i] != min_vals[i]:
                normalized_val = (val - min_vals[i]) / (max_vals[i] - min_vals[i])
            else:
                normalized_val = 0.0
            normalized_features.append(normalized_val)

        normalized_element = Element(
            features=normalized_features,
            element_id=element_obj.id,
            categorical_data=element_obj.categorical_data
        )
        normalized_element.is_centroid = element_obj.is_centroid

        normalized_map[el_id] = {
            'element_obj': normalized_element,
            'cluster_id': data['cluster_id']
        }

    logger.log("Features normalizadas para KNN.")
    return normalized_map

def find_k_nearest_neighbors(
    all_elements_map: Dict[str, Dict[str, Any]],
    target_features: List[float],
    k: int
) -> List[Tuple[str, float, Dict[str, Any]]]:
    if not all_elements_map:
        logger.log("Nenhum elemento disponível para KNN.")
        return []

    distances: List[Tuple[str, float, Element]] = []

    # Verificacao basica de dimençao
    any_el_data = next(iter(all_elements_map.values()))
    expected_dim = len(any_el_data['element_obj'].features)
    if len(target_features) != expected_dim:
        logger.log(f"Incompatibilidade dimensional: esperado {expected_dim}, obtido {len(target_features)}.")
        return []

    # Calcula distancias
    for el_id, data_dict in all_elements_map.items():
        element_obj: Element = data_dict['element_obj']
        try:
            dist = Cluster.euclidean_distance(target_features, element_obj.features)
            distances.append((el_id, dist, element_obj))
        except ValueError as e:
            logger.log(f"Erro ao calcular distancia para elemento {el_id}: {e}")
            continue

    distances.sort(key=lambda x: x[1])

    # Prepara resultado
    results: List[Tuple[str, float, Dict[str, Any]]] = []
    for el_id, dist, el_obj in distances[:k]:
        neighbor_info = {
            'element_id': el_id,
            'features': el_obj.features,
            'cluster_id': all_elements_map[el_id]['cluster_id'],
            'categorical_data': el_obj.categorical_data
        }
        results.append((el_id, dist, neighbor_info))

    logger.log(f"Encontrados {len(results)} vizinhos mais próximos.")
    return results

def predict_class_with_knn(k_nearest_neighbors_info: List[Tuple[str, float, Dict[str, Any]]]) -> Optional[str]:
    if not k_nearest_neighbors_info:
        return None

    neighbor_cluster_ids = [info_tuple[2]['cluster_id'] for info_tuple in k_nearest_neighbors_info if info_tuple[2]]
    if not neighbor_cluster_ids:
        logger.log("Nenhum cluster ID encontrado entre os vizinhos.")
        return None

    majority_vote = Counter(neighbor_cluster_ids).most_common(1)
    prediction = majority_vote[0][0] if majority_vote else None
    logger.log(f"Predição KNN: {prediction}")
    return prediction

# Exemplo de uso
def create_bitcoin_example():
    logger.log("=== EXEMPLO: Análise de Padrões de Alta do Bitcoin ===")

    manager = ClusterManager(dispersion_threshold=500000000.0)

    initial_bitcoin_data = [
        [15.0, 2023.0, 45000.0, 2000000000.0],
        [180.0, 2023.0, 65000.0, 4000000000.0]
    ]

    manager.create_initial_clusters(initial_bitcoin_data)
    logger.log("Clusters iniciais criados para análise de Bitcoin:")
    for cid in manager.clusters:
        details = manager.get_cluster_details(cid)
        logger.log(f"  {details}")

    novos_dados = [
        ([30.0, 2023.0, 47000.0, 2200000000.0], {"tendencia": "alta", "volatilidade": "baixa"}),
        ([45.0, 2023.0, 48500.0, 2400000000.0], {"tendencia": "alta", "volatilidade": "media"}),
        ([175.0, 2023.0, 63000.0, 3800000000.0], {"tendencia": "alta", "volatilidade": "alta"}),
        ([190.0, 2023.0, 67000.0, 4200000000.0], {"tendencia": "alta", "volatilidade": "alta"}),
        ([200.0, 2023.0, 52000.0, 2800000000.0], {"tendencia": "correcao", "volatilidade": "media"})
    ]

    logger.log("\nAdicionando novos dados de Bitcoin:")
    for features, categorical in novos_dados:
        el_id, cluster_id = manager.add_new_record_to_system(features, categorical)
        logger.log(f"  Elemento {el_id} -> Cluster {cluster_id}")

    logger.log("\nEstado final dos clusters:")
    all_details = manager.get_all_cluster_details()
    for cid, details in all_details.items():
        if details:
            logger.log(f"  Cluster {cid}: {details['elements'].__len__()} elementos, dispersão: {details['dispersion']:.2f}")

    logger.log("\n=== Demonstração KNN ===")
    normalized_elements = normalize_features_for_knn(manager.all_elements_map)

    target_point = [60.0, 2023.0, 50000.0, 2600000000.0, 0.0, 1.0]
    neighbors = find_k_nearest_neighbors(normalized_elements, target_point, k=3)

    logger.log(f"Vizinhos mais próximos para {target_point[:4]}:")
    for i, (el_id, dist, info) in enumerate(neighbors):
        logger.log(f"  {i+1}. Elemento {el_id} (distância: {dist:.4f}) - Cluster: {info['cluster_id']}")

    prediction = predict_class_with_knn(neighbors)
    logger.log(f"Predição de cluster para novo ponto: {prediction}")

    logger.log("\nMapeamentos categóricos criados:")
    for category, mapping in manager.categorical_mappings.items():
        logger.log(f"  {category}: {mapping}")

if __name__ == "__main__":
    # Executa exemplo
    create_bitcoin_example()

    logger.log("\n=== SCRIPT FINALIZADO ===")