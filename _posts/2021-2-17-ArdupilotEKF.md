---
layout: post
title: Làm thế nào Ardupilot chạy được EKF với 24 states?
date:   2021-2-17
categories: [STM32,Embedded]
published: true
mathjax: true
background: '/PATH_TO_IMAGE'
---

Bài này nói về phương pháp xử lý bộ EKF của ardupilot.

Bộ EKF này đóng vài trò navigation cũng như AHRS.

Phần cứng ardupilot phần lớn là STM32F4, với lượng flash giới hạn, khả năng xử lý cũng giới hạn. Do đó phương pháp tối ưu hóa của họ là làm hoàn toàn trên matlab, sau đó sinh code, kể cả phép nghịch đảo ma trận.

Github ở đây: https://github.com/priseborough/InertialNav

Dĩ nhiên việc nghịch đảo ma trận 24*24 trên vi điều khiển là một việc khá là mệt, do đó để giảm thiểu khối lượng tính toán, tác giả bộ lọc sử dụng phương pháp phần chia các vector measurement theo từng trục đo và thực hiện nhiều bước fusion.

Phương pháp này làm giảm nhẹ gánh nặng phải nghịch đảo ma trận lớn, tuy nhiên cũng làm cho bộ lọc mất ổn định hơn.

24 trạng thái được ước lượng là:

![image.png]({{ site.baseurl }}/images/ekf_new/state.PNG)

 4 Quaternion, 3 Gyroscope bias, 3 Accelerometer bias, 3 giá trị Magnet North East Down, 3 giá trị Magnet Bias, 3 giá trị tọa độ North East Down, 3 giá trị vận tốc North East Down. 

 Thay vì dùng thư viện ma trận, tác giả Ardupilot chọn phương pháp viết các phương trình trạng thái ra, tuyến tính hóa bằng hàm Jacobian của Matlab, sau đó dùng hàm subexpr để Optimize lại phương trình.

 Tuy còn nhiều cái chưa hiểu lắm, nhưng mà cơ bản là bằng cách này tác giả không cần thực hiện phép nghịch đảo 24*24, và code hoàn toàn tự sinh ra. 

File sinh các phươn trình này là file 


![image.png]({{ site.baseurl }}/images/ekf_new/statem.PNG)

Trừ giá trị gyro đi bias, cái này thì y chang như EKF mình hay làm :3

![image.png]({{ site.baseurl }}/images/ekf_new/quat.PNG)

Dưới đây là phương trình prediction, giá trị quaternion mới = 1/2*quaternion cũ * gyro

Phương trình sau đó là tích phân giá trị gia tốc để ra được vận tốc theo 3 trục NED, gia tốc này được lấy có lẽ từ giá trị accelerometer chiếu lên NED, với hệ máy bay thì chỉ cần thế này, còn xe cộ thì chắc là phải trừ đi g để ra được gia tốc ngoại lực.

![image.png]({{ site.baseurl }}/images/ekf_new/quat.PNG)

Magnetometer được đối xử khá là đặc biệt khi mình nó có 3 trục và 6 trạng thái cần ước lượng.

![image.png]({{ site.baseurl }}/images/ekf_new/magnet1.PNG)

Thay vì phương pháp chọn 1 trục magnet như mình làm, tác giả ước lượng magnet theo giá trị ở cả 3 trục NED, không biết là lúc đầu calib lại hay là dựa trên bản đồ từ, nhưng ưu điểm là cách này giúp ước lượng declination và inclination tốt hơn 

![image.png]({{ site.baseurl }}/images/ekf_new/magnet2.png)

Phần này mình chưa hiểu rõ lắm cách xử lý yaw của bộ lọc.

Tóm lại là các phương trình EKF như vậy

Bước tuyến tính hóa:

Bước này dùng công cụ symbolic để tính ra được jacobian của ma trận 
![image.png]({{ site.baseurl }}/images/ekf_new/jacobi1.PNG)

Hàm OptimiseAlgebra sẽ tìm các biến chung trong phương trình symbolic để thay thế, từ đó giảm khối lượng tính toán.

Tiếp theo là phần tuyến tính hóa ma trận quan sát:

Tác giả tiếp tục tính Jacobian để tìm ra ma trận H dựa trên phương trình trạng thái. 
![image.png]({{ site.baseurl }}/images/ekf_new/jacobi1.PNG)

Như vậy thay vì dùng giấy tính, việc thiết kế EKF này có thể hoàn toàn tự động luôn, vì có hàm sinh code C ở cuối. 

Có một số điểm mình chưa hiểu lắm:

  1. Trong các bước fusion này, mình chưa hiểu rõ vai trò correction của accelerometer đóng góp vào đo đạc góc lắm.
  2. Tương tự như vậy với magnetometer, phương pháp cập nhật yaw cũng chưa rõ lắm, vì trong các phương trình measurement không thấy đề cập đến cách cập nhật yaw. Có thể vì phần code này chỉ dành cho mục đích navigation chăng?. 
  3. Chắc chắn phải có GPS hay một sensor absolute nào đó tác động vào để correct lại hết chứ nhỉ? :v

Các phần code có thể tham khảo từ model simulink của bộ lọc trong github đã có link ở trên 

![image.png]({{ site.baseurl }}/images/ekf_new/model.PNG)

Tóm lại: Cách thiết kế khá là chuyên nghiệp :3 có thể mình sẽ có thử nghiệm thiết kế bộ lọc theo phong cách này
