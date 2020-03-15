---
layout: post
title: Kiến trúc Mobilenet
date:   2018-6-28 12:23:17 -0500
categories: [Computer Vision,deep learning]
published: true
mathjax: true
image:
  background: /images/post5/3.jpg
---

Thực sự mấy hôm nay thử cài lại mobilenet mà khó khăn quá :( viết note tổng hợp lại vậy :( (hong phải lấy keras ra đâu :3)
# 1.Depthwise Separate Convolutional


Convolutional là phép toán phổ biến trong ngành điện, và Yann LeCun áp dụng nó vào trong xử lý ảnh và ra kết quả tuyệt vời, Mục đích của nó là trích xuất được các đặc trưng cần thiết và giảm số chiều.
Cách làm việc của lớp Convolutional:

![an image alt text]({{ site.baseurl }}/images/post5/1.jpg "CNN layer")

_Convolutional hoạt động_

Mỗi Convolutional filter là một ma trân có kích thức nhỏ (tách đặc trưng local), khi hoạt động mỗi filter này thực hiện một phép nhân với một khu vực có kích thức bằng với kích thước như kernel. Cách thực hiện có thể thấy ở trên hình: Nhân từng số ở cùng vị trí rồi cộng lại với nhau. Mỗi kernel đuợc quét qua cả hình với mỗi lần quét gọi là strides, tức strides=1 thì kernel di chuyển 1 hàng or cột 1 lần. Còn ở những khu vực ngoài cùng nếu thiếu thì đơn giản là bổ sung thêm 1 hoặc 2 hàng số 0 nếu như kernel không chia hết. Về phần này ta có khái niệm padding VALID và padding SAME trong tensorflow (keras). Với Valid thì nếu kernel không chia hết thì đơn giản là skip đoạn thừa ra. Còn SAME là sẽ tìm cách thêm một số hàng (cột) toàn 0 để không mất feature.

Một ví dụ về convolutional là bộ lọc Sobel

![an image alt text]({{ site.baseurl }}/images/post5/2.jpg "\Sobel Filter")

Bây giờ xét về độ phức tạp tính toán:

![an image alt text]({{ site.baseurl }}/images/post5/3.jpg "Convolution layer")

_Một layer của convolution neural network_

Trong hình là một kernel. Ta thấy rằng 1 hình ảnh có số chiều là (H,W,N) với H là cao, W là rộng, H là sâu (3 trong trường hợp ảnh RGB). 1 kernel điển hình gồm (K,K,N) với K là độ lớn kernel, N là chiều sâu ảnh.

Chiều ra trong hình được tính là độ rộng ra  $$= (W - K+2P)/S+1)$$  (với P là padding, S là strides)
Tương tự với chiều cao. Độ phức tạp tính toán với một phép Convolutional trên 1 filter sẽ là $$K * K $$ ,gọi D là chiều của feature map sau khi quét qua tất cả H và W. Vậy thì số phép nhân sẽ là $$ D * D * K^2 $$ , quét qua chiều sâu N là $$ D * D * K^2 *N $$. Nhưng ta lại có nhiều kernel, vì vậy số phép nhân thực tế sẽ là $$ M*(D^2*K^2*N) $$ với M là số kernel.
Khá là lớn. ý tưởng của depthwise separate convolutional là chia phép convolutional thành 2 phần:
## 1.Depthwise convolutional

   Bước này thực chất là thay vì sử dụng 1 convolutional layer cho 3 kênh RGB, ta dùng một filter cho 1 kênh riêng biệt. Vì vậy số phép nhân sẽ là $$ 1*(K^2 * D^2) *N $$ 
Sau phép convolution này ta sẽ có số chiều output như là phép convolution cũ, với số lượng layer sẽ là N với N là số chiều ảnh.

![an image alt text]({{ site.baseurl }}/images/post5/4.jpg "depthwise convolution layer")

Giải thích về Depthwise Separate Convolutional
## 2.Pointwise Convolutional 
Sau phép Depthwise, ta thực hiện phép convolution 1*1 để trích M đặc trưng ra từ N lớp Convolutional đã tính ở trên:

![an image alt text]({{ site.baseurl }}/images/post5/5.jpg "convolutional 1*1")

_Phép convolutional 1*1_

Thực chất nếu như chỉ có 1 layer thì ta có thể thấy 1*1 convolutional chỉ giống như nhân 1 số vào ảnh, nhưng nếu có nhiều hơn 1 layer, thì tác dụng của nó giống như 1 neural net trong 1 neural net vậy.

{% include youtube.html id="vcp0XvDAX68" %}

Để hiểu hơn có thể đọc paper Network in Network.
Tóm lại ta sau phép Pointwise ta nhận được số output tensor y chang như phép convolutional cổ điển.
Số phép nhân là: $$ D^2* M * N $$  (Quét qua D*D, nhân với M filter, N chiều)
Do đó tổng cộng lại ta có số phép nhân là: $$ 1*(K^2 * D^2) * N + D^2 * M* N = N*D^2(M+K^2) $$
Chia cho số phép nhân trong convolutional truyền thống:

$$ (N*D^2(M+K^2)) / (M*(D^2*K^2*N)) = 1/M + 1/(K^2) $$

Cho 64 filter, filter (3*3) thì ta được khoảng 0.126 tức giảm được tầm 8 lần :3



Depthwise được dùng trong các mạng: One model to learn them all (nghe meme vãi :)) ), Xception, và mobilenet.

# 2.Kiến trúc mobilenet

Mấy cái tinh hoa nhất thì nói rùi :)) thôi thì nói ít vậy :))

![an image alt text]({{ site.baseurl }}/images/post5/6.jpg "Mobilenet structure")

_Kiến trúc_

Mobilenet sử dụng rất nhiều depthwise separete convolutional layer để giảm thiểu số parameter khoảng 9 lần, phù hợp cho các ứng dụng mobile, mà vẫn giữ được độ chính xác chỉ thua 1% so với convolutional bình thường, theo paper thì khoảng 95% số phép nhân nằm ở thao tác 1*1 convolution. Ngoài ra các tác giả cũng thêm hằng số alpha để có thể giảm thiểu số chiều của mạng một cách nhẹ nhàng cho các ứng dụng cần mức nhỏ :3, nói chung paper này còn khá mới (1704), tiềm năng của nó khá là lớn, khi nén lại thì mobilenet chỉ khoảng 0.92 mb!! :3 quá là nhỏ :3, và dùng nó để object detection cũng khá là phê :3 máy i5 đời 2 như tớ cũng xài được :)), cơ mà không hiểu sao bữa làm xoài detector mà kết quả như **** ấy :)))

![an image alt text]({{ site.baseurl }}/images/post5/7.jpg "Xoài detector")

_Xoài detector của tớ :))_

Demo thì trên mạng nhiều mà thấy nhiều cái cũng ảo diệu quá :)))

{% include youtube.html id="06DGkJ8wYK0" %}


{% include youtube.html id="9qPHL0hbTHE" %}

_Demo trên S7 ;)_
{% include youtube.html id="PcCc_wROVPE" %}

paper:https://arxiv.org/pdf/1704.04861.pdf
Bản implement chính thức có thể tìm tại github của tensorflow
*Bài viết tham khảo rất nhiều nguồn internet