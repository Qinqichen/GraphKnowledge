# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 17:02:29 2021

@author: QinQichen
"""

import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import copy

class App:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def create_friendship(self, person1_name, person2_name):
        with self.driver.session() as session:
            # Write transactions allow the driver to handle retries and transient errors
            result = session.write_transaction(
                self._create_and_return_friendship, person1_name, person2_name)
            for record in result:
                print("Created friendship between: {p1}, {p2}".format(
                    p1=record['p1'], p2=record['p2']))

    @staticmethod
    def _create_and_return_friendship(tx, person1_name, person2_name):

        # To learn more about the Cypher syntax,
        # see https://neo4j.com/docs/cypher-manual/current/

        # The Reference Card is also a good resource for keywords,
        # see https://neo4j.com/docs/cypher-refcard/current/

        query = (
            "CREATE (p1:Person { name: $person1_name }) "
            "CREATE (p2:Person { name: $person2_name }) "
            "CREATE (p1)-[:KNOWS]->(p2) "
            "RETURN p1, p2"
        )
        result = tx.run(query, person1_name=person1_name, person2_name=person2_name)
        try:
            return [{"p1": record["p1"]["name"], "p2": record["p2"]["name"]}
                    for record in result]
        # Capture any errors along with the query and data for traceability
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def find_person(self, person_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_and_return_person, person_name)
            for record in result:
                print("Found person: {record}".format(record=record))

    @staticmethod
    def _find_and_return_person(tx, person_name):
        query = (
            "MATCH (p:Person) "
            "WHERE p.name = $person_name "
            "RETURN p.name AS name"
        )
        result = tx.run(query, person_name=person_name)
        return [record["name"] for record in result]
    
    
    def find_all_n(self,n):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_all_n, n)
            
            print(result)
            
            print('---------------')
            
            for record in result:
                
                print("Found: {record}".format(record=record))
    
    @staticmethod
    def _find_all_n(tx,n):
        query = (
            'MATCH (n:Employee) RETURN n LIMIT 25'
        )
        result = tx.run(query)
        
        return [record["n"] for record in result]
    
    @staticmethod
    def _executeQuery(tx,queryText):
        query = (
            queryText
        )
        result = tx.run(query)
        
        print(queryText+"执行完成")
        
        return [ [record["n"],record['r'],record['m']] for record in result ]
        
        
        
    def executeQuery(self,queryText):
        with self.driver.session() as session:
            result = session.read_transaction(self._executeQuery, queryText)
            return result
        
    @staticmethod
    def b_executeQueryRead(tx,queryText):
        
        query = (queryText)
        
        result = tx.run(query)
        
        return [ record for record in result ]
        
    def executeQueryRead(self,queryText):
        
        with self.driver.session() as session:
            
            result = session.read_transaction(self.b_executeQueryRead,queryText)
            
            return result
    
    
    
    
    
    

if __name__ == "__main__":
    # See https://neo4j.com/developer/aura-connect-driver/ for Aura specific connection URL.
    scheme = "neo4j"  # Connecting to Aura, use the "neo4j+s" URI scheme
    host_name = "127.0.0.1"
    port = 7687
    url = "{scheme}://{host_name}:{port}".format(scheme=scheme, host_name=host_name, port=port)
    user = "neo4j"
    password = "qinqichen"
    app = App(url, user, password)
    # app.create_friendship("Alice", "David")
    
    relation = ["match(p:演员{name:'周星驰'}),(m:演员家属{name:'凌宝儿'}) CREATE (p)-[母亲:mather]->(m)",
            "match(p:演员{name:'周星驰'}),(m:演员家属{name:'周文姬'}) CREATE (p)-[姐姐:sister]->(m)",
            "match(p:演员{name:'周星驰'}),(m:演员家属{name:'周星霞'}) CREATE (p)-[妹妹:sister]->(m)",
            "match(p:演员{name:'周星驰'}),(m:演员{name:'莫文蔚'}) CREATE (p)-[前女友:girlfriend]->(m)",
            "match(p:演员{name:'周星驰'}),(m:路人{name:'于文凤'}) CREATE (p)-[前女友:girlfriend]->(m)",
            "match(p:演员{name:'周星驰'}),(m:演员{name:'朱茵'}) CREATE (p)-[前女友:girlfriend]->(m)",
            "match(p:演员{name:'周星驰'}),(m:演员{name:'罗慧娟'}) CREATE (p)-[前女友:girlfriend]->(m)",
            "match(p:演员{name:'周星驰'}),(m:演员{name:'徐娇'}) CREATE (p)-[义女:daughter]->(m)",
            "match(p:演员{name:'周星驰'}),(m:歌手{name:'张学友'}) CREATE (p)-[好友:friend]->(m)",
            "match(p:演员{name:'周星驰'}),(m:演员{name:'吴孟达'}) CREATE (p)-[搭档:friend]->(m)",
            "match(p:演员{name:'周星驰'}),(m:演员{name:'温兆伦'}) CREATE (p)-[搭档:friend]->(m)",
            "match(p:演员{name:'周星驰'}),(m:导演{name:'王晶'}) CREATE (p)-[搭档:friend]->(m)",
            "match(p:歌手{name:'张学友'}),(m:演员{name:'罗美薇'}) CREATE (p)-[妻子:wife]->(m)",
            "match(p:演员{name:'罗美薇'}),(m:歌手{name:'张学友'}) CREATE (p)-[丈夫:husband]->(m)",
            "match(p:演员家属{name:'张瑶萱'}),(m:演员{name:'罗美薇'}) CREATE (p)-[母亲:mather]->(m)",
            "match(p:演员家属{name:'张瑶萱'}),(m:歌手{name:'张学友'}) CREATE (p)-[父亲:father]->(m)",
            "match(p:演员{name:'罗美薇'}),(m:演员家属{name:'张瑶萱'}) CREATE (p)-[女儿:daughter]->(m)",
            "match(p:歌手{name:'张学友'}),(m:演员家属{name:'张瑶萱'}) CREATE (p)-[女儿:daughter]->(m)",
            "match(p:歌手{name:'张学友'}),(m:歌手{name:'梅艳芳'}) CREATE (p)-[好友:friend]->(m)",
            "match(p:歌手{name:'梅艳芳'}),(m:歌手{name:'张学友'}) CREATE (p)-[好友:friend]->(m)",
            "match(p:歌手{name:'张学友'}),(m:演员{name:'周星驰'}) CREATE (p)-[好友:friend]->(m)",
            "match(p:歌手{name:'张学友'}),(m:演员家属{name:'张作琪'}) CREATE (p)-[父亲:father]->(m)",
            "match(p:演员家属{name:'张作琪'}),(m:歌手{name:'张学友'}) CREATE (p)-[儿子:son]->(m)",
            "match(p:歌手{name:'黄贯中'}),(m:演员{name:'朱茵'}) CREATE (p)-[妻子:wife]->(m)",
            "match(p:演员{name:'朱茵'}),(m:歌手{name:'黄贯中'}) CREATE (p)-[丈夫:husband]->(m)"
        ]   
    
    # for q in relation:
    #     app.executeQuery( q)
    
    qText = "MATCH (n)-[r]->(m) RETURN n,r,m"
    
    result = app.executeQuery(qText)
    
    httpResult = {
        
            "nodes":[],
            "links":[],
            "categories":[]
         }
    
    node = {
              "id": "0",
              "name": "Myriel",
              "value": 1,
              "category": 0
            }
    link =  {
               "source": "1",
               "target": "0",
               "label":{
                   "show":'true',
                   'formatter':''
                   },
               "lineStyle": {
                   "curveness": 0.2
                }
            }
    category = {
                "name": "类目0"
              }
    
    nodeSet = set()
    linkSet = set()
    cateSet = set()
    cateDict = dict()
    
    for record in result:
        
        nodeSet.add(record[0])
        nodeSet.add(record[2])
        linkSet.add(record[1])
        
        for n in record[0].labels :
            cateSet.add(n)
        for n in record[2].labels :
            cateSet.add(n)
        
    i = 0 
    for n in cateSet:
        cateDict[n] = i 
        i+=1
        category['name'] = n
        httpResult['categories'].append(category.copy())
        
    
    for n in nodeSet :
        node['id'] = str(n.id)
        node['name'] = n['name']
        
        for i in n.labels:
            node['category'] = cateDict[i]
        # print(node)
        httpResult['nodes'].append(node.copy())
        
    for n in linkSet :
        
        link['source'] = str(n.nodes[0].id)
        link['target'] = str(n.nodes[1].id)
        link['label']['formatter'] = n.type
        
        # print(link)
        httpResult['links'].append(copy.deepcopy(link))
    
    print(httpResult)
    
    # app.find_person("Alice")
    # app.find_all_n(25)
    app.close()

