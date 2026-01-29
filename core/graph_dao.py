from neo4j import GraphDatabase
import logging
import csv
import os

logger = logging.getLogger(__name__)

class GraphDAO:
    def __init__(self, uri, user, password, use_mock=True, local_csv_path=None):
        self.use_mock = use_mock
        self.driver = None
        self.local_csv_path = local_csv_path
        self._csv_cache = {}
        
        if not self.use_mock:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                # Verify connection
                self.driver.verify_connectivity()
                logger.info("Connected to Neo4j database.")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}. Switching to Mock mode.")
                self.use_mock = True
        else:
            logger.info("GraphDAO initialized in Mock mode.")

    def close(self):
        if self.driver:
            self.driver.close()

    def query_entity_relation(self, entity, relation, time=None):
        """
        Query the graph for a specific relation of an entity, optionally filtered by time.
        """
        if self.driver:
            query = """
            MATCH (h:Entity {name: $entity})-[r:RELATION {name: $relation}]->(t:Entity)
            WHERE $time IS NULL OR r.time = $time
            RETURN t.name as tail
            """
            try:
                with self.driver.session() as session:
                    result = session.run(query, entity=entity, relation=relation, time=time)
                    return [record["tail"] for record in result]
            except Exception as e:
                logger.error(f"Cypher query failed: {e}")
                return []

        local = self._local_query(entity, relation, time)
        if local:
            return local

        if self.use_mock:
            return self._mock_query(entity, relation, time)
        
        return []

    def _normalize_time_prefix(self, time):
        if not time:
            return None
        digits = ''.join(ch for ch in str(time) if ch.isdigit())
        if len(digits) >= 8:
            return digits[:8]
        if len(digits) >= 6:
            return digits[:6]
        if len(digits) >= 4:
            return digits[:4]
        return None

    def _local_query(self, entity, relation, time):
        if not self.local_csv_path or not os.path.exists(self.local_csv_path):
            return []

        time_prefix = self._normalize_time_prefix(time)
        cache_key = (str(entity), str(relation), time_prefix)
        if cache_key in self._csv_cache:
            return self._csv_cache[cache_key]

        results = set()
        best_time = None
        try:
            with open(self.local_csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('head') != str(entity):
                        continue
                    if row.get('relation') != str(relation):
                        continue
                    row_time = row.get('time') or ''
                    if time_prefix:
                        if len(time_prefix) == 8:
                            if row_time != time_prefix:
                                continue
                        else:
                            if not row_time.startswith(time_prefix):
                                continue
                    tail = row.get('tail')
                    if tail:
                        if time_prefix and len(time_prefix) < 8:
                            if best_time is None or row_time > best_time:
                                best_time = row_time
                                results = {tail}
                            elif row_time == best_time:
                                results.add(tail)
                        else:
                            results.add(tail)
        except Exception as e:
            logger.error(f"Local CSV query failed: {e}")
            self._csv_cache[cache_key] = []
            return []

        out = sorted(results)
        self._csv_cache[cache_key] = out
        return out

    def _mock_query(self, entity, relation, time):
        """
        Returns hardcoded data for demonstration.
        """
        logger.info(f"Mock Query: {entity} - {relation} - {time}")
        
        # Scenario 1: Guizhou Moutai Independent Director
        if "贵州茅台" in str(entity) and "独立董事" in str(relation):
            return ["陆金海", "姜国华", "郭田勇"]
            
        # Scenario 2: Major Shareholder
        if "大股东" in str(relation):
            return ["中国贵州茅台酒厂(集团)有限责任公司"]
            
        return []
