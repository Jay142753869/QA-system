from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class GraphDAO:
    def __init__(self, uri, user, password, use_mock=True):
        self.use_mock = use_mock
        self.driver = None
        
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
        if self.use_mock:
            return self._mock_query(entity, relation, time)
        
        # Real Cypher implementation (Template)
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
            
        return ["暂无数据 (Mock Data)"]
