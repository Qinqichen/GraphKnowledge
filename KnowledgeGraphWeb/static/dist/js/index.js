

var machineMessage = "";
$.get("/web/static/component/machineMessage.html",function (html) {
    machineMessage =  $(html);
});

$("#submitQuestion").off("click").on("click",function () {

    console.log("点击事件");
    let question = $("#inputQuestion").val();
    console.log(question);
    if (question =="" || question == null || question == undefined)
        return false;


    let contentList = $("#questionContentList");
    if (question === "clear"){
        contentList.empty();
        $("#inputQuestion").val("");
        return false;
    }
    $.get("/web/static/component/userMessage.html",function (data) {
        let message = $(data);
        message.find("p:eq(0)").text(question);
        message.find("small").text(new Date().format("yy-MM-dd hh:mm:ss"));
        contentList.append(message);
        let contentScroll = document.getElementById("questionContentList");
        contentScroll.scrollTop = contentScroll.scrollHeight;
        
        
        $.ajax({
            type: "POST",
            url: "/knowledgeGraphServer/getAnswer",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({question:question}),
            success: function (result) {
                let machineResponse = machineMessage.clone();
                let resultMessage = ""
                console.log(result);

                
                if(result['error'].length == 0){
                
                    resultMessage = result['view'][0]['question'] + " 的答案为： " + result['view'][0]['answer'];
                }
                else{
                    resultMessage = "错误信息： "  + result['error'][0]['description'];
                    
                }
                
                
                
                
                
                
                
                if( question == "张学友" || question == "周星驰" ){
                
                    var myChart = echarts.getInstanceByDom(document.getElementById('main'));
                    
                    $.getJSON('/KnowledgeModel/getNodesByNodesName/'+question,function(graph){
                    
                        console.log(graph)
                        option = {
                                    series: [
                                        {
                                            data: graph.nodes,
                                            links: graph.links,
                                            categories: graph.categories
                                        }
                                    ]
                                };
                            
                                myChart.setOption(option);
                            
                    
                        });
                    
                
                
                }
                else{
                    var myChart = echarts.getInstanceByDom(document.getElementById('main'));
                    
                    $.getJSON('/KnowledgeModel/getIndexShowGraphData',function(graph){
                    
                        console.log(graph)
                        option = {
                                    series: [
                                        {
                                            data: graph.nodes,
                                            links: graph.links,
                                            categories: graph.categories
                                        }
                                    ]
                                };
                            
                                myChart.setOption(option);
                            
                    
                        });
                
                }
                
                
                
                
                
                
                
                
                
                
                
                machineResponse.find("p:eq(0)").text(resultMessage);
                machineResponse.find("small").text(new Date().format("yy-MM-dd hh:mm:ss"));
                contentList.append(machineResponse);
                let contentScroll = document.getElementById("questionContentList");
                contentScroll.scrollTop = contentScroll.scrollHeight;
            }
        })
        
        
    });

    $("#inputQuestion").val("");

});





















