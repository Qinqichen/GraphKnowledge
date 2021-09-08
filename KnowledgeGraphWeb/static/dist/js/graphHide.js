var ROOT_PATH = '/KnowledgeModel/getIndexShowGraphData';

var chartDom = document.getElementById('main');
var myChart = echarts.init(chartDom);
var option;

myChart.showLoading();
$.getJSON(ROOT_PATH , function (graph) {
    myChart.hideLoading();

    console.log(graph)
    
    option = {
        tooltip: {},
        legend:{
            top:30,
            data:graph.categories.map(function (a) {
                return a.name;
            })
        },
        title:{
            text:"文物信息可视化",
            left:"center"
        },
        series: [
            {
                name: 'Les Miserables',
                type: 'graph',
                layout: 'force',
                data: graph.nodes,
                links: graph.links,
                categories: graph.categories,
                draggable: true,
                roam: true,
                label: {
                    show: true,
                    formatter: '{b}'
                },
                labelLayout: {
                    hideOverlap: true
                },
              /***  scaleLimit: {
                    min: 1
                    max: 2
                },
               ****/
                lineStyle: {
                    color: 'source',
                    curveness: 0.3
                }
            }
        ]
    };

    myChart.setOption(option);
});

option && myChart.setOption(option);
