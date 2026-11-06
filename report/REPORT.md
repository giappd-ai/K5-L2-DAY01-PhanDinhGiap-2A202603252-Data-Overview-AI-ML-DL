# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab  ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này đại diện cho cả bức ảnh.
Lý do: Đây là tác vụ Phân loại ảnh (Image Classification). Mô hình nhận đầu vào là một bức ảnh và trả ra phân phối xác suất cho toàn bộ ngữ cảnh/chủ thể bao trùm của ảnh đó, chứ không xác định vị trí tọa độ (bounding box) cho từng vật thể riêng lẻ bên trong.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp này đi kèm với Taxonomy/Checkpoint (ImageNet-1K được huấn luyện sẵn trên checkpoint yolo11n-cls.pt), chứ mô hình hoàn toàn không tự nghĩ ra tên nhãn.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
class_id: Là định danh số nguyên cố định, giúp máy tính/hệ thống lập trình xử lý chuẩn xác không bị ảnh hưởng bởi ngôn ngữ.
class_name: Giúp con người (annotator, engineer) đọc hiểu trực quan lớp đó là gì.
taxonomy_name: Để định vị tập nhãn chuẩn đang sử dụng (tránh nhầm lẫn giữa tập ImageNet-1K với COCO-80, Pascal VOC, v.v.).
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Trong tác vụ Phân loại ảnh (Image Classification), đích đến cuối cùng là gán một nhãn duy nhất đại diện cho toàn bộ bức ảnh. Khi bức ảnh có bối cảnh phức tạp chứa nhiều thực thể khác nhau (như ảnh traffic có cả xe buýt, xe taxi, người đi bộ, lòng đường), tài liệu hướng dẫn (guideline) bắt buộc phải quy định các điểm sau để điều phối người gán nhãn (annotator):

Quy tắc ưu tiên chủ thể chính (Dominant Class/Salient Object Rule):
Hướng dẫn annotator chọn nhãn của thực thể chiếm diện tích bề mặt lớn nhất, nằm ở trung tâm hoặc là trọng tâm chính thu hút thị giác của bức ảnh.
Hệ thống cấp bậc ưu tiên (Priority Hierarchy):
Định nghĩa rõ thứ tự ưu tiên của các nhóm nhãn khi chúng xuất hiện đồng thời. Ví dụ: "Sinh vật sống (Người/Động vật) > Phương tiện giao thông > Đồ vật tĩnh > Cảnh nền đất đá".
Quy tắc xử lý cảnh hỗn hợp (Compound/Scene Rule):
Nếu ảnh biểu thị một hoạt động tập thể hoặc cảnh quan chung, guideline phải yêu cầu annotator chọn nhãn mô tả toàn cảnh (như street hoặc traffic_scene nếu có trong Taxonomy) thay vì chọn nhãn của từng phương tiện đơn lẻ như cab hay bus.
Quy định xử lý độ không chắc chắn (Uncertainty & Escalation Guide):
Khi không thể phân định rõ ràng chủ thể nào là chính, annotator phải tuân thủ quy tắc đồng thuận nhóm hoặc đưa lên mức cao hơn (Escalation) cho Lab Coach/QC chứ không tự ý đoán mò.
6. Vì sao không được lấy `score` của model làm ground truth?
Tuyệt đối không được lấy score (độ tin cậy) của mô hình làm nhãn chuẩn (ground truth) vì những lý do cốt lõi sau:

Bản chất của Score (Model Confidence):
score (ví dụ: 0.510915 cho lớp cab) chỉ là một giá trị xác suất toán học xuất phát từ hàm Softmax ở lớp cuối cùng của mô hình. Nó chỉ thể hiện mức độ tự tin của mô hình dựa trên các trọng số toán học hiện tại của nó, không đại diện cho sự thật khách quan.
Mô hình có thể sai sót hoặc bị lệch (Bias & Errors):
Mô hình YOLO11-cls chưa hoàn hảo; nó có thể đưa ra độ tin cậy rất cao cho một dự đoán sai do hiện tượng nhiễu ảnh hoặc do tập dữ liệu huấn luyện của nó bị lệch. Nếu lấy score làm chuẩn, ta sẽ trực tiếp đưa những sai lệch và lỗi của AI vào tập dữ liệu chuẩn.
Giới hạn của Taxonomy ban đầu:
Như đã thấy trong ảnh traffic, mô hình dự đoán lớp cab đạt hạng 1 chỉ vì bộ nhãn ImageNet-1K không có lớp tổng quát nào đúng hơn (như "cảnh tắc đường"). Điểm số 51% này chỉ là sự lựa chọn tốt nhất trong số các nhãn có sẵn của mô hình, hoàn toàn không phải nhãn đúng mà con người mong muốn gán cho bức ảnh giao thông đó.
Ground Truth yêu cầu tính chính xác tuyệt đối (Gold Standard):
Ground Truth phải là sự thật tuyệt đối do con người kiểm chứng theo một bộ quy tắc logic nhất quán (guideline). Đây là dữ liệu dùng để huấn luyện và đánh giá mô hình, do đó nó phải là tiêu chuẩn vàng sạch sẽ nhất, không được phép phụ thuộc vào xác suất của một mô hình chưa hoàn thiện.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58  ; `bbox_height`=279.68 
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Dựa vào tọa độ hộp giới hạn [385.33, 69.24, 498.92, 348.92] của đối tượng person ở trên:

Góc trên - bên trái (Top-Left point): Nằm tại điểm cách biên bên trái của bức ảnh là 385.33 pixel và cách biên phía trên của bức ảnh là 69.24 pixel.
Góc dưới - bên phải (Bottom-Right point): Nằm tại điểm cách biên bên trái của bức ảnh là 498.92 pixel và cách biên phía trên của bức ảnh là 348.92 pixel.
Kích thước thực tế: Vật thể này có chiều ngang trải rộng 113.58 pixel và chiều cao kéo dài 279.68 pixel, tạo thành một khung hình chữ nhật bao bọc vừa vặn cơ thể người đứng bếp.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ngưỡng 0.20 (Hạ thấp ngưỡng): Có 17 vật thể được phát hiện.
Danh sách nhãn phát hiện: ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle'].
Ngưỡng 0.60 (Thắt chặt ngưỡng): Chỉ còn 6 vật thể được phát hiện.
Danh sách nhãn phát hiện: ['person', 'bowl', 'bowl', 'oven', 'oven', 'person'].
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng lọc từ 0.60 xuống 0.20, số lượng hộp xuất hiện tăng vọt (từ 6 lên 17 hộp) dẫn đến nhiều hộp bị lỗi hoặc trùng lặp. Người Reviewer / QC phải thực hiện thêm các nghiệp vụ:

Xóa bỏ (Delete/Reject): Loại bỏ các hộp bị trùng lặp trên cùng một đối tượng (Double bounding boxes) và các hộp gán sai mục tiêu do nhiễu môi trường (False Positives - ví dụ: chiếc thìa spoon hoặc cái bát bowl bị nhận diện sai lệch do độ tin cậy quá thấp).
Chỉnh sửa phân loại (Re-classify): Đổi lại nhãn đúng cho các hộp bị gán sai lớp.
Nắn chỉnh biên (Refine Box): Điều chỉnh lại mép hộp cho ôm sát hoàn toàn đối tượng thực tế.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Khi hạ ngưỡng lọc từ 0.60 xuống 0.20, số lượng hộp xuất hiện tăng vọt (từ 6 lên 17 hộp) dẫn đến nhiều hộp bị lỗi hoặc trùng lặp. Người Reviewer / QC phải thực hiện thêm các nghiệp vụ:

Xóa bỏ (Delete/Reject): Loại bỏ các hộp bị trùng lặp trên cùng một đối tượng (Double bounding boxes) và các hộp gán sai mục tiêu do nhiễu môi trường (False Positives - ví dụ: chiếc thìa spoon hoặc cái bát bowl bị nhận diện sai lệch do độ tin cậy quá thấp).
Chỉnh sửa phân loại (Re-classify): Đổi lại nhãn đúng cho các hộp bị gán sai lớp.
Nắn chỉnh biên (Refine Box): Điều chỉnh lại mép hộp cho ôm sát hoàn toàn đối tượng thực tế.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Vật thể minh chứng trên ảnh:
Chiếc lò nướng (oven) ở rìa dưới bên trái bức ảnh bị cắt mép (Truncated) chỉ còn nhìn thấy một phần.
Các chiếc xoong/nồi (bowl/pot) treo ở giàn phía trên bên trái bị che khuất lẫn nhau rất nhiều (Occluded).
Phương án giải quyết: Đưa vào quy tắc trong Guideline chứ không sử dụng Escalation.
Lý do: Hiện tượng che khuất (Occlusion) và bị cắt rìa ảnh (Truncation) là những kịch bản vô cùng phổ biến và xảy ra liên tục trong hầu hết mọi bài toán thị giác máy tính thực tế. Để quy trình dán nhãn diễn ra trôi chảy trên diện rộng với hàng trăm annotator, guideline bắt buộc phải chuẩn hóa sẵn quy tắc xử lý cho trường hợp này (ví dụ: 'Vẫn vẽ hộp bao quanh phần nhìn thấy được nếu vật thể bị che khuất dưới 80%, vẽ sát vào rìa ảnh đối với các phần bị cắt'). Chúng ta chỉ dùng Escalation (báo lên Lab Coach) khi gặp những dị vật cực kỳ kỳ lạ, không thể xác định nổi nhãn danh mục hoặc ảnh bị hỏng cấu trúc nghiêm trọng.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-011 ; `class_name`=spoon ; `score`=0.359147 ; số điểm polygon=`polygon_point_count`=37 ; 3–5 điểm đầu của `polygon_xy`=   
      [
        529.0,
        66.0
      ],
      [
        529.0,
        72.0
      ],
      [
        528.0,
        73.0
      ],
      [
        528.0,
        74.0
      ],
      [
        527.0,
        75.0
      ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Hộp giới hạn (Bounding Box): Chỉ cung cấp vùng không gian hình chữ nhật thô sơ bao ngoài vật thể. Nó bị lẫn rất nhiều pixel nền (background) không thuộc về đối tượng (ví dụ: khoảng trống giữa hai chân hoặc hai bên sườn của người đứng bếp).
Đa giác (Polygon / Mask): Cung cấp cấu trúc hình học chi tiết, độ cong, và biên giới chính xác của đối tượng đến từng pixel. Nó phân tách hoàn toàn các điểm ảnh thực sự thuộc về người đứng bếp khỏi phần nền (như mảng tường bếp, dụng cụ treo phía sau), giúp mô hình học được hình dáng thực tế của vật thể thay vì chỉ học một vùng ảnh vuông
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id (ví dụ: kitchen-011): Là một mã định danh duy nhất được gán tạm thời cho một thực thể độc lập cụ thể trong bức ảnh hiện tại trong phiên xử lý đó.
Vì sao không phải Class ID? Class ID dùng chung cho cả một nhóm đối tượng cùng loại (ví dụ: tất cả mọi người trong ảnh đều có Class ID = 0). Trong khi đó, hai người khác nhau đứng cạnh nhau sẽ có hai instance_id riêng biệt.
Vì sao không phải Tracking ID? Tracking ID là mã định danh xuyên suốt dùng để theo vết, nối khớp một thực thể qua các khung hình liên tiếp của một video. Trong khi đó, instance_id ở đây chỉ tồn tại độc lập trong duy nhất một bức ảnh tĩnh này.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
"Quy tắc Vẽ biên đa giác Ôm khít pixel (Pixel-Perfect Boundary Rule): Người gán nhãn phải đặt các điểm neo (vertices) bám sát theo đúng đường biên phân chia thực tế của vật thể. Khoảng cách tối đa giữa đường biên đa giác và rìa thực tế của vật thể không được vượt quá 1 pixel. Không được cắt lẹm vào cơ thể vật thể và tuyệt đối không được lấy lấn ra ngoài phần nền sau. Tại các vùng có độ cong lớn (như vai, đầu), phải tăng mật độ điểm neo để đảm bảo đường đa giác mô tả trung thực hình dáng vật thể."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Điểm giao cắt mờ trên ảnh: Vùng tạp dề màu trắng của người đứng bếp chồng lấn lên chiếc bàn gỗ chứa bột ở phía trước, hoặc phần cánh tay phải của người bị che khuất một phần bởi các dụng cụ treo xung quanh.
Phương án giải quyết: Sử dụng Guideline có sẵn để xử lý trực tiếp thay vì Escalation.
Lý do: Đây là trường hợp chồng lấn và che khuất một phần (Occlusion) hoàn toàn kinh điển trong các bài toán thực tế. Guideline chuẩn đã quy định rõ: 'Khi vật thể A chồng lên vật thể B, phần biên đa giác của vật thể A (ở phía trước) sẽ được vẽ trọn vẹn, còn vật thể B (ở phía sau) chỉ được vẽ phần diện tích thực tế nhìn thấy được, tuyệt đối không tự ý vẽ lấn phần giả định bị che khuất'. Do quy tắc đã quá rõ ràng nên Annotator tự tin thực hiện theo tài liệu hướng dẫn mà không cần tốn thời gian leo thang hỏi Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Là một nhãn đơn duy nhất (Single categorical label) gán cho toàn bộ bức ảnh (ví dụ: cab, minibus, streetcar).  | Phân loại ImageNet-1K đưa ra dự đoán lớp hàng đầu cho ảnh traffic là cab với score thấp (0.510915), trong khi đây là cảnh giao thông đường bộ hỗn hợp nhiều xe buýt và xe khách chứ không chỉ có taxi. | Xem toàn bộ bức ảnh, đối chiếu danh sách lớp phân loại (Taxonomy) để chọn ra một nhãn bao quát đúng nhất bản chất trung tâm của cảnh. | Đối chiếu nhãn Annotator chọn với ảnh thực tế để phát hiện sai sót phân loại gán nhãn chủ quan, hoặc kiểm tra xem nhãn được chọn có nằm trong quy định phân loại không. |
| Phát hiện vật thể | Là danh sách các cặp nhãn lớp kèm tọa độ hộp giới hạn dạng số pixel: [class_name, [x_min, y_min, x_max, y_max]] (ví dụ: [person, [385.33, 69.24, 498.92, 348.92]]). |  Ở ngưỡng thấp 0.20, mô hình sinh ra trùng lặp nhiều hộp trên cùng đối tượng (như 3 hộp bowl đè lên nhau) hoặc nhận diện sai các vùng nhiễu mờ như các muôi thìa treo tường thành spoon với điểm tin cậy thấp.| Vẽ các hộp giới hạn dạng chữ nhật ôm khít biên ngoài cùng của từng đối tượng đơn lẻ và chọn phân loại lớp tương ứng cho hộp theo tài liệu hướng dẫn. | Kiểm tra độ ôm khít của các hộp giới hạn (Bounding Box), xóa bỏ các hộp bị trùng lặp, điều chỉnh hộp bị lệch biên, và bổ sung các vật thể bị Annotator bỏ sót. |
| Instance segmentation | Là danh sách tọa độ chuỗi điểm đa giác ôm sát biên đối tượng cùng định danh duy nhất: [instance_id, class_name, [[x1, y1], [x2, y2], ...]]. | Biên mờ mịt giữa dĩa/bát nấu ăn và mặt bàn gỗ chứa bột màu trắng làm đa giác phân đoạn của chiếc bàn gỗ dining table-011 hoặc chiếc bát bowl-009 bị lem nhem, răng cưa và chồng chéo không rõ ràng. | Dùng công cụ đa giác (Polygon Tool) chấm các điểm neo chính xác bám sát biên pixel thực tế của từng đối tượng độc lập để tạo mặt nạ phủ (Mask) hoàn chỉnh. | Kiểm tra độ tinh xảo của các đường biên đa giác, đảm bảo đường vẽ không bị răng cưa cẩu thả, không cắt lẹm vào đối tượng, và xử lý đúng quy tắc chồng lấn ở các vị trí tiếp giáp. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?

"Trong suốt bài lab này, chỉ làm việc và chạy thực nghiệm trên ba bức ảnh công khai có sẵn từ tập dữ liệu COCO, tuyệt đối không tải lên hoặc xử lý bất kỳ hình ảnh cá nhân, thông tin nhạy cảm hay dữ liệu nội bộ chưa được cấp phép nào."
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

"Nếu phát hiện hình ảnh hoặc dữ liệu không hợp lệ, nằm ngoài phạm vi nghiên cứu hoặc chứa thông tin nhạy cảm, tôi sẽ ngay lập tức dừng công việc và báo cáo trực tiếp cho Lab Coach để nhận chỉ dẫn xử lý tiếp theo."

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
