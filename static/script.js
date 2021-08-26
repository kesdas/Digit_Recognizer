var canvas = document.getElementById("paint");
var context = canvas.getContext("2d");
//var width = canvas.width;
//var height = canvas.height;
//var curX, curY, prevX, prevY;
var hold = false;
context.lineWidth = 13;
var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint;

//
//canvas.addEventListener("mousedown", mouseDown, false);
//        canvas.addEventListener("mousemove", mouseXY, false);
//        document.body.addEventListener("mouseup", mouseUp, false);
//
//
//        //For mobile
//        canvas.addEventListener("touchstart", mouseDown, false);
//        canvas.addEventListener("touchmove", mouseXY, true);
//        canvas.addEventListener("touchend", mouseUp, false);
//        document.body.addEventListener("touchcancel", mouseUp, false);
//
//    function draw() {
//        for (var i = 0; i < clickX.length; i++) {
//            context.beginPath();                               //create a path
//            if (clickDrag[i] && i) {
//                context.moveTo(clickX[i - 1], clickY[i - 1]);  //move to
//            } else {
//                context.moveTo(clickX[i] - 1, clickY[i]);      //move to
//            }
//            context.lineTo(clickX[i], clickY[i]);              //draw a line
//            context.stroke();                                  //filled with "ink"
//            context.closePath();                               //close path
//        }
//    }
//
//    canvas.ontouchstart = function(e) {
//        if (e.touches) e = e.touches[0];
//        return false;
//    }
//
//    function addClick(x, y, dragging) {
//        clickX.push(x);
//        clickY.push(y);
//        clickDrag.push(dragging);
//    }
//
//    function mouseXY(e) {
//       var touches = e.touches || [];
//       var touch = touches[0] || {};
//       if (paint) {
//                addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop  , true);
//                draw();
//             }
//    }
//
//    function mouseUp() {
//      paint = false;
//    }
//
//    function mouseDown(e)
//    {
//      var mouseX = e.pageX - this.offsetLeft;
//            var mouseY = e.pageY - this.offsetTop;
//            paint = true;
//            addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop );
//            draw();
//    }

function add_pixel(){
    context.lineWidth += 4;
}

function reduce_pixel(){
    if (context.lineWidth == 1){
        context.lineWidth = 1;
    }
    else{
        context.lineWidth -= 4;
    }
}


function reset(){
    clickX = new Array();
    clickY = new Array();
    clickDrag = new Array();
    context.clearRect(0, 0, canvas.width, canvas.height);
}

// pencil tool

function pencil(){

    canvas.onmousedown = function(e){
        curX = e.clientX - canvas.offsetLeft;
        curY = e.clientY - canvas.offsetTop;
        hold = true;

        prevX = curX;
        prevY = curY;
        context.beginPath();
        context.moveTo(prevX, prevY);
    };

    canvas.onmousemove = function(e){
        if(hold){
            curX = e.clientX - canvas.offsetLeft;
            curY = e.clientY - canvas.offsetTop;
            draw();
        }
    };

    canvas.onmouseup = function(e){
        hold = false;
    };

    canvas.onmouseout = function(e){
        hold = false;
    };

    function draw(){
        context.lineTo(curX, curY);
        context.stroke();
//        canvas_data.pencil.push({ "startx": prevX, "starty": prevY, "endx": curX, "endy": curY, "thick": context.lineWidth, "color": context.strokeStyle });
    }
}

function save(){
    var image = canvas.toDataURL();
    var pred = document.getElementById('prediction').textContent;
    var prob = document.querySelectorAll('#img_div img')[0].src;
    var guess_box = $('#guess_box').is(':checked')?$('#guess_box').val():false

    $.post("/", { save_image: image, prediction: pred , guess: guess_box, prob_img: prob}, function
    () {
    location.href = "/";
});
//    alert('Result has been saved');
}

function submit(){
    var image = canvas.toDataURL();

    $.post("/prediction", { save_image: image }, function(data) {
    var cont = document.getElementById("prediction");
    var img = document.getElementById("show_graph")
        cont.textContent=data[0];
        img.src = data[1];
        cont.insertAdjacentElement('afterend', img);
    });
}