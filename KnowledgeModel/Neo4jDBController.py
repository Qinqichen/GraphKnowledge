# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:49:50 2021

@author: QinQichen
"""

from .Neo4jAPP import App
import copy



class Neo4jControllerClass:
    
    
    def __init__(self, url, user, password):
        self.app = App(url, user, password)
        self.httpResponse = {
            "nodes":[],
            "links":[],
            "categories":[]
         }
        self.node = {
              "id": "-1",
              "name": "",
              "value": -1,
              "category": -1
            }
        self.link =  {
               "source": "-1",
               "target": "-1",
               "label":{
                    "show":'true',
                    'formatter':''
                   },
                # "lineStyle": {
                #     "curveness": 0.2
                # }
            }
        self.category =  {
                "name": ""
              }
    
    
    def getCategoriesSet(self,categories = []):
        
        cSet = set()
        
        for c in categories:
            
            cSet.add(c)
            
            
        return cSet
    
    def getCategoiesDict(self,categoriesSet = set()):
        
        cDict = dict()
        
        i = 0 
        
        for c in categoriesSet:
            
            cDict[c] = i 
            i += 1 
        
    
        return cDict

    
    def transNodes(self,nodes = [] , cateDict = dict()):
        
        nodeSet = set()
        
        for n in nodes:
            nodeSet.add(n)
        
        node = copy.deepcopy(self.node)
        
        nList = []
        
        for n in nodeSet :
            if n['name'] == None :
                continue
            node['id'] = str(n.id)
            node['name'] = n['name']
            
            for i in n.labels:
                node['category'] = cateDict[i]
                
            nList.append(copy.deepcopy(node))
            
            
        return nList
            
        
        pass
    def transLinks(self,links = []):
        
        linkSet = set()
        
        for l in links:
            
            linkSet.add(l)
        
        link = copy.deepcopy(self.link)
        
        lList = []
        
        for l in linkSet :
        
            link['source'] = str(l.nodes[0].id)
            link['target'] = str(l.nodes[1].id)
            link['label']['formatter'] = l.type
            
            lList.append(copy.deepcopy(link))
            
    
        
        return lList
    
    def transCategories(self,cDict = dict()):
        
        category = copy.deepcopy(self.category)
        
        cList = []
        
        for c in cDict:
            
            category['name'] = c
            
            cList.append(copy.deepcopy(category))
            
    
        return cList
    
    
    def n_r_m_2_httpResponse(self,result):
        
        nodes = []
        links = []
        categories = []
        
        for r in result:
            nodes.append(r['n'])
            nodes.append(r['m'])
            links.append(r['r'])
            
            for i in r['n'].labels:
                categories.append(i)
            for i in r['m'].labels:
                categories.append(i)
        
        cDict = self.getCategoiesDict(self.getCategoriesSet(categories))
        
        nList = self.transNodes(nodes,cDict)
        lList = self.transLinks(links)
        cList = self.transCategories(cDict)
        
        httpResponse = copy.deepcopy(self.httpResponse)
        
        httpResponse['nodes'] = nList
        httpResponse['links'] = lList
        httpResponse['categories'] = cList
        
        return httpResponse
    
    
    def getNodesAndLinksByNodesName(self,nameNodes):
        
        result = self.app.executeQueryRead("match(n{name:'"+ nameNodes +"'}) -[r]-> (m) return n,r,m")
        
        
        httpResponse = self.n_r_m_2_httpResponse(result)
        
        return httpResponse
    

    def getIndexShowData(self):
        
        result = self.app.executeQueryRead("MATCH (n)-[r]->(m) RETURN n,r,m")
        
        nodes = []
        links = []
        categories = []
        
        for r in result:
            nodes.append(r['n'])
            nodes.append(r['m'])
            links.append(r['r'])
            
            for i in r['n'].labels:
                categories.append(i)
            for i in r['m'].labels:
                categories.append(i)
        
        cDict = self.getCategoiesDict(self.getCategoriesSet(categories))
        
        nList = self.transNodes(nodes,cDict)
        lList = self.transLinks(links)
        cList = self.transCategories(cDict)
        
        httpResponse = copy.deepcopy(self.httpResponse)
        
        httpResponse['nodes'] = nList
        httpResponse['links'] = lList
        httpResponse['categories'] = cList
        
        print(httpResponse)
        
        
        
        return httpResponse

    def close(self):
        
        self.app.close()



if __name__ == "__main__":
    
    scheme = "neo4j"  # Connecting to Aura, use the "neo4j+s" URI scheme
    host_name = "127.0.0.1"
    port = 7687
    url = "{scheme}://{host_name}:{port}".format(scheme=scheme, host_name=host_name, port=port)
    user = "neo4j"
    password = "qinqichen"
    app = Neo4jControllerClass(url, user, password)
    
    
    # app.getIndexShowData()
    
    r = app.getNodesAndLinksByNodesName('张学友')
    
   
    app.close()



























