

var machineMessage = "";
$.get("./component/machineMessage.html",function (html) {
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
        return false;
    }
    $.get("./component/userMessage.html",function (data) {
        let message = $(data);
        message.find("p:eq(0)").text(question);
        message.find("small").text(new Date().format("yy-MM-dd hh:mm:ss"));
        contentList.append(message);
        let contentScroll = document.getElementById("questionContentList");
        contentScroll.scrollTop = contentScroll.scrollHeight;
        $.get("./component/test",function (result) {
            let machineResponse = machineMessage.clone();
            machineResponse.find("p:eq(0)").text(result);
            machineResponse.find("small").text(new Date().format("yy-MM-dd hh:mm:ss"));
            contentList.append(machineResponse);
            let contentScroll = document.getElementById("questionContentList");
            contentScroll.scrollTop = contentScroll.scrollHeight;
        });
    });


});





















