---
layout: post
title: So sánh 3 thuật toán sensor fusion trên IMU
date:   2020-2-26 
categories: [EKF,Quaternion]
published: true
mathjax: true
---

Mình viết cái Kalman đã lâu, nhưng gần đây nghỉ Tết mới có thời gian coi lại và phát hiện ra một rổi bug :3 thế là viết lại, và sau khi viết lại + ăn ngủ, chơi Civilization, đây là kết quả so sánh 3 thuật toán - complementary, madgwick và kalman trong điều kiện gần-như-được-soft-mounting-trên-drone (nói chung là thêm mấy cái đệm cao su thôi :)) ) và kéo throttle ầm ầm lên cho có tí nhiếu 

# Độ chính xác 
Điều kiện thử nghiệm: 3 thuật toán này được cài trên freeRTOS, task lấy data raw highPriority 1000Hz, task filter 100Hz, trước khi đưa data vào các bộ lọc chính là một bộ lọc trung vị, task thứ 3 là task send data, do giới hạn của bên nhận nên chỉ chạy ở 33Hz để đảm bảo tính real time cho data gửi về.

Điều kiện thử nghiệm: 3 thuật toán này được cài trên freeRTOS, task lấy data raw highPriority 1000Hz, task filter 100Hz, trước khi đưa data vào các bộ lọc chính là một bộ lọc trung vị, task thứ 3 là task send data, do giới hạn của bên nhận nên chỉ chạy ở 33Hz để đảm bảo tính real time cho data gửi về.

Throttle tầm 85%, drone đặt nằm yên.


![an image alt text]({{ site.baseurl }}/images/postCom/1.jpg "Compare")

* 3 hình trên cùng lần lượt là roll pitch yaw  là lọc bù có alpha = 0.99, chú ý mình chưa cài AHRS mà chỉ là filter IMU

* 3 hình tiếp theo là Madgwick beta = 0.01

* hình cuối là Kalman quaternion cơ bản, với x 7 trạng thái, 4 quaternion và 3 bias, Q cho 3 bias là 1e-4, R cho 3 trục accelerometer là 0.001.Bên phải là data từ một board flight control sử dụng betaflight, board này được hard mounting 

![an image alt text]({{ site.baseurl }}/images/postCom/2.jpg "Compare")

![an image alt text]({{ site.baseurl }}/images/postCom/3.jpg "Compare")

*Khác biệt về góc giữa 3 bộ lọc là do có sai sót khi calib data, việc này không đáng kể khi trừ offset*

Kết quả trên cho thấy là độ chính xác của lọc bù và Madgwick trong khoảng (-1 đến 1) độ, trong khi đó Kalman là [-2 2] độ. Đây là kết quả khá đáng thất vọng :3
Trong khi đó bé fc mua cũng không khác hơn, thực ra là do bị dính chặt :v, roll thâm chí nhảy từ khoảng 6.5 - 3.8 độ. “sad vl”
Vì vậy mình đã chỉnh lại R Kalman là 0.1 và lọc bù alpha 0.998:

![an image alt text]({{ site.baseurl }}/images/postCom/4.jpg "Compare")

Lần chạy này ở throttle max, các bộ lọc đều sai lệch trong khoảng [-1 1] độ :3 (nói chung là cũng không tốt nhắm ) 
Ở throttle thấp hơn có vẻ là lỗi ít hơn tầm +-0.5 độ cho đội Kalman

Tóm lại: drone kiểu này sẽ lạng rất nhiều, chắc chắn không kéo throttle 100% được.
# Về thời gian chạy 

Phần cứng sử dụng: Stm32f405rgt6, MPU9250, tần số I2C 400kHz
 Để đo thời gian chạy mình sủ dụng 1 timer có prescaler = 167, internal clock = 168Mhz, counter = 10000, tức nó có thể đo được tối đa là 0.01s:

![an image alt text]({{ site.baseurl }}/images/postCom/5.jpg "Compare")

Kalman:




![an image alt text]({{ site.baseurl }}/images/postCom/6.jpg "Compare")

![an image alt text]({{ site.baseurl }}/images/postCom/7.jpg "Compare")

Lấy trung bình 10 mẫu được 1834 count, tức là 0.001834s -> tần số là 545.256 Hz
Madgwick:


![an image alt text]({{ site.baseurl }}/images/postCom/9.jpg "Compare")

![an image alt text]({{ site.baseurl }}/images/postCom/8.jpg "Compare")



Lấy max là 10 count -> thời gian thực hiện là 0.0001 s :”> (nà ní ) nói chung là rất nhanh.
Lọc bù :


![an image alt text]({{ site.baseurl }}/images/postCom/10.jpg "Compare")


3 count :3 -> 0.000003 s :3 (cái này thì đúng)
Tóm lại : không biết có gì sai sót không nhưng sao nó lại như vậy 

P/s Kalman hàng nhà trồng (matlab gen code như cái **** ấy, chả chạy được) nên chả tối ưu gì cả, inverse xài “kết liễu Gauss" (Gaussian Elimination trong đstt ) nên ma trận rất dễ ăn nan :v. Mong cao nhân nào chỉ giáo cho drone em bay

Link: https://github.com/DangLamTung/drone-RTOS

# P/s của ps: hết áy náy, game tiếp

