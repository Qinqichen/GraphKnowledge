Date.prototype.format = function (fmt) {
    var o = {
        "M+": this.getMonth() + 1,                   //月份
        "d+": this.getDate(),                        //日
        "h+": this.getHours(),                       //小时
        "m+": this.getMinutes(),                     //分
        "s+": this.getSeconds(),                     //秒
        "q+": Math.floor((this.getMonth() + 3) / 3), //季度
        "S": this.getMilliseconds()                  //毫秒
    };

    //  获取年份
    // ①
    if (/(y+)/i.test(fmt)) {
        fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
    }

    for (var k in o) {
        // ②
        if (new RegExp("(" + k + ")", "i").test(fmt)) {
            fmt = fmt.replace(
                RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[k]).length)));
        }
    }
    return fmt;
}

function toastMessage(message,title,time) {

    time = time == null ? "刚刚":time;
    title = title == null ? "提醒":title;

    var html = "<div class=\"position-fixed bottom-0 right-0 p-3\" style=\"z-index: 5; right: 0; bottom: 0;width: 300px;\">\n" +
        "                        <div id=\"liveToastOfTool\" class=\"toast hide\" role=\"alert\" aria-live=\"assertive\" aria-atomic=\"true\" data-delay=\"2000\" style=\"z-index: 2000\">\n" +
        "                            <div class=\"toast-header\">\n" +
        "                                <!--<img src=\"https://gitee.com/qinqichen/img-store/raw/master/img/20210612110847.png\" class=\"rounded mr-2 h-100\" alt=\"...\">-->\n" +
        "                                <i class=\"fa fa-commenting-o\" aria-hidden=\"true\" class=\"rounded\"></i>\n" +
        "                                <strong class=\"mr-auto ml-2\">提醒</strong>\n" +
        "                                <small>刚刚</small>\n" +
        "                                <button type=\"button\" class=\"ml-2 mb-1 close\" data-dismiss=\"toast\" aria-label=\"Close\">\n" +
        "                                    <span aria-hidden=\"true\">&times;</span>\n" +
        "                                </button>\n" +
        "                            </div>\n" +
        "                            <div class=\"toast-body\">\n" +
        "                                Hello, world! This is a toast message.\n" +
        "                            </div>\n" +
        "                        </div>\n" +
        "                    </div>";

    $("body").append(html);

    $("#liveToastOfTool").find("smell").text(time);
    $("#liveToastOfTool").find(".toast-body").text(message);
    $("#liveToastOfTool").find("strong").text(title);

    $("#liveToastOfTool").toast("show");

}




/**
 * 有问题，待改正
 * @param href
 */
// jQuery.extend({
//     getUrlAllAttrObject:function (href) {
//         var all = href.split('?')[1].split('&');
//         var result = {};
//         for (var i = 0 ; i < all.length ; i ++ ){
//             result.add(all[i].split("=")[0],all[i].split("=")[0]);
//         }
//         return result;
// //     }
// // });
//
// function formArrayToJson(array) {
//     var json ;
//
//     for (var key in array){
//         json.key = array.key;
//     }
//     return json;
// }