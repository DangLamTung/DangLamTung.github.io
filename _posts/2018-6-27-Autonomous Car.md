---
layout: post
title: Xe tự lái có chọi lại ninja Lead ??? :D ???
categories: Autonomous Car, deep learning
published: true
mathjax: true
---

# 1.Xe tự lái là cái chi chi :#
Search google là biết rùi đó :)))))))))))))))))))

![an image alt text]({{ site.baseurl }}/images/post4/1.jpg "Xe tự lái")

Tất nhiên không phải nói cái này rồi :)) mà là autonomous car:

![an image alt text]({{ site.baseurl }}/images/post4/2.jpg "Xe tự lái")

_Vừa đọc sách vừa lái xe :3 phê quá còn gì_
Vậy xe tự lái thì có gì hot ??? :3 ??? máy bay có thể tự lái được, thuyền cũng tự lái được, nhưng ô tô tự lái có thể đem lại nhiều điều hay ho :3. Ví dụ như ở Mĩ thì xe tải chính là huyết mạch của kinh tế Mĩ

![an image alt text]({{ site.baseurl }}/images/post4/3.jpg "Xe tự lái")

Mà xe thì tất nhiên phải có người lái, việc ăn ngủ hay thuê người lái nói chung là tổn phí, mà vận chuyển bằng tàu hay máy bay thì không linh động hay chi phí cao hơn nên không thể phát triển như đường bộ được. Vì thế ta cần xe tự lái.
Khác với máy bay hay tàu thủy có xác suất tai nạn thấp (cứ thử tưởng tượng 2 con tàu biển đâm nhau thử :))) ) thì đường bộ với đặc điểm nhỏ hẹp thực sự rất dễ tai nạn, nên hệ thống tự lái của ô tô khá là phức tạp so với máy bay hay tàu thủy.

![an image alt text]({{ site.baseurl }}/images/post4/4.jpg "Xe tự lái")

Nhưng khó không có nghĩa không làm được :3, nhất là ta đang trong thời đại bùng nổ của deep learning :3
# 2.Cách tiếp cận bài toán xe tự lái

![an image alt text]({{ site.baseurl }}/images/post4/5.jpg "Xe tự lái")

## Phương pháp tiếp cận bài toán xe tự lái của Ndivia

Đầu tiên và cơ bản nhất để làm một chiếc xe tự lái, ta phải xác định được phần đường mình cần đi, phần này ta sẽ tách phần đường mình cần ra.

![an image alt text]({{ site.baseurl }}/images/post4/6.jpg "Xe tự lái")

## Nhận dạng phần đường

Sau đó ta cần phải tìm góc của vô lăng dựa trên phần đường cần tìm, demo dưới đây dùng CNN:

![an image alt text]({{ site.baseurl }}/images/post4/7.jpg "Xe tự lái")

_Demo tìm góc của vô lăng_

Vậy là xe đã có thể “lái” được. Tuy nhiên chỉ đơn giản là “lái” được. Cái mà chúng ta cần ví dụ như là gọi một xe taxi tự lái đến đón ta ở nhà và đưa ta đến đâu đó thì chưa đủ. Thực chất cái khó của xe tự lái nằm ở cách làm thế nào để biến mấy món đồ chơi thành cái gì đó sinh lợi nhuận được. Vì vậy chúng ta mới cần:



## Lidar và sensor fusion

![an image alt text]({{ site.baseurl }}/images/post4/7.jpg "Xe tự lái")

Trong nhiều điều kiện, cảm biến quang là không đủ để đưa ra nhận định chính xác về môi trường, vì vậy cần phải có nhiều các cảm biến khác để nâng cao độ an toàn. Ví dụ như lidar để đo khoảng cách chính xác, không phụ thuộc vào ánh sáng (vì là sóng ngắn), còn radar lại tốt hơn trong điều kiện mưa, vì vậy phối hợp 3 loại cảm biến này là quan trọng và nói chung là khó :3 riêng cái Kalman thôi học cũng muốn sml :))))))

## Path-planing và GPS:

![an image alt text]({{ site.baseurl }}/images/post4/8.jpg "Xe tự lái")

Ví dụ như đi trên đường có những thanh niên leader đi chặn đầu thì phải đi ra sao, đến nhà crush thì đi đường nào :3 những cái đó là phần này phải giải quyết, tưởng tượng đương đầu với leader là thấy mệt rồi :))))))) phần này nói chung là mệt, phải kết hợp nhiều kiến thức xử lý tín hiệu, máy học,......... 

## Object detection:

Lidar hay radar chỉ cho ta biết dạng của vật thể chứ không thể cho ta biết nó là cái gì, người vật,.... nên rất cần một hệ thống Object Detection ổn ổn để xe có thể né ra :3

![an image alt text]({{ site.baseurl }}/images/post4/9.jpg "Xe tự lái")

Ngoài ra còn có các vấn đề như giao thức điều khiển,.... noi chung đây là một bài toán khá là khó.
## 2. Làm thử 1 cái đê :3

Tuy là khó, nhưng không phải là không làm được :3

Nếu chỉ làm một chiếc chơi thì không cần care mấy cái thứ phức tạp trên làm gì :3 ta chỉ cần care về cái cách xử lý CNN thôi :3

Dưới đây là mô hình CNN của Ndivia :3 (pilotnet):

![an image alt text]({{ site.baseurl }}/images/post4/10.jpg "Xe tự lái")

_Ndivia pilotnet_

Còn đây là mô hình inception 

![an image alt text]({{ site.baseurl }}/images/post4/11.jpg "Xe tự lái")

_Inception_

Đó :3 căn bản là ez hơn nhiều :3
Về cách mà CNN này hoạt động cũng đơn giản: đưa hình vào CNN và nó predict góc của vô lăng:

![an image alt text]({{ site.baseurl }}/images/post4/12.jpg "Xe tự lái")

_Hình ảnh thể hiện những nơi được tập trung để đưa ra góc quay_

Qúa trình thu data cũng đơn giản: người ta theo quay video đường người lái và góc vô lăng, sau đó lưu lại, rồi train mạng theo đó. Lưu ý là góc quay đã được đưa về một khoảng xác định trong khoảng của neuron output.
Còn về hiệu quả :3 sau đây là xe của Ndivia:

{% include youtube.html id="NJU9ULQUwng" %}

Khá là OK ;) tuy nhiên nói luôn là không chọi lại mấy chị Leader đâu :) xe ở Mĩ mà chạy VN 100% là sml :))))))))). Vấn đề chính là do CNN cần nhiều data, và gặp data chưa thấy bao h là tiêu. Do đó thằng có nhiều data nhất là thằng thắng. Đó cũng là lý do kỉ nguyên AI này google nắm trùm :3
Còn một số link tham khảo khá hay:

https://devblogs.nvidia.com/deep-learning-self-driving-cars/ link chính chủ Ndivia

https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013 Chứng chỉ của udacity, 800 đô, cơ mà code thì có đăng github:

https://github.com/udacity/self-driving-car

Và một demo pilotnet bằng Tensorflow:

https://github.com/SullyChen/Autopilot-TensorFlow

Xe đồ chơi của một bác người Việt :3 chạy khá ngon :#

https://github.com/experiencor/self-driving-toy-car

Nhiêu đó là đủ làm một chiếc chạy chơi rùi ha :)) mà muốn đi Cuộc đua số thì còn thêm nhiều cơ :)))))))))))))