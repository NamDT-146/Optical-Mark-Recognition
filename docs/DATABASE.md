Đề xuất Triển khai CP4: Xử lý Đa luồng & Cơ sở dữ liệu Room

Tài liệu này mô tả chi tiết cách cấu hình CameraX phân luồng kết hợp với JNI C++ và lược đồ cơ sở dữ liệu SQLite (sử dụng thư viện Room) dựa trên yêu cầu của UI Design (omr_design_prompt.md).

1. Xử lý Đa luồng (Background Thread) cho CameraX & JNI

Để đạt được FPS > 1 và không gây treo UI, nguyên tắc cốt lõi là tuyệt đối không gọi hàm JNI trên Main Thread. Chúng ta sẽ sử dụng ImageAnalysis của CameraX kết hợp với Kotlin Coroutines hoặc ExecutorService.

1.1 Cơ chế "Bỏ qua Frame" (Frame Dropping / Backpressure)

Camera có thể trả về 30-60 frames mỗi giây, nhưng thuật toán C++ của bạn có thể mất 100ms - 300ms để xử lý 1 frame. Nếu đẩy toàn bộ frame vào JNI, ứng dụng sẽ bị tràn bộ nhớ (OOM) hoặc giật lag.

Đề xuất implementation cho Analyzer:

class OMRImageAnalyzer(
    private val onResult: (OMRResult) -> Unit
) : ImageAnalysis.Analyzer {

    // Cờ đánh dấu đang xử lý để bỏ qua các frame đến sau
    private var isProcessing = AtomicBoolean(false)

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        // Nếu đang xử lý frame trước đó, đóng frame hiện tại ngay lập tức
        if (isProcessing.get()) {
            imageProxy.close()
            return
        }

        isProcessing.set(true)

        // Chuyển việc xử lý sang Background Thread (CPU optimized)
        CoroutineScope(Dispatchers.Default).launch {
            try {
                // 1. Chuyển đổi ImageProxy sang định dạng OpenCV cần (ví dụ: NV21 byte array hoặc Bitmap)
                val bitmapData = ImageUtils.convertYUV420ToBitmap(imageProxy)
                
                // 2. Gọi hàm JNI C++ (Hàm này chạy trên Background thread, an toàn)
                // C++ pipeline: Lọc nhiễu -> Tìm 4 anchors -> Perspective Transform -> Chấm điểm
                val resultJson = OMRNativeLib.processImage(bitmapData)
                
                // 3. Parse kết quả từ C++
                val omrResult = parseResult(resultJson)

                // 4. Trả kết quả về Main Thread để cập nhật UI (AR Overlay / Toast)
                withContext(Dispatchers.Main) {
                    onResult(omrResult)
                }
            } catch (e: Exception) {
                Log.e("OMRAnalyzer", "Lỗi xử lý ảnh", e)
            } finally {
                // Giải phóng frame và mở cờ cho frame tiếp theo
                imageProxy.close()
                isProcessing.set(false)
            }
        }
    }
}


1.2 Cấu hình CameraX Executor

Khi khởi tạo ImageAnalysis, hãy gán cho nó một thread pool riêng biệt để tách hoàn toàn khỏi luồng UI:

val imageAnalysis = ImageAnalysis.Builder()
    .setTargetResolution(Size(1280, 720)) // Khuyến nghị độ phân giải vừa đủ để tối ưu C++
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .build()

// Sử dụng Executor có 1 luồng duy nhất để tránh xung đột JNI
val analyzerExecutor = Executors.newSingleThreadExecutor()
imageAnalysis.setAnalyzer(analyzerExecutor, OMRImageAnalyzer { result -> 
    // Update UI (Vẽ khung xanh, hiện điểm số)
    updateAROverlay(result)
    // Lưu vào Database bất đồng bộ
    saveResultToDatabase(result)
})


2. Lược đồ Cơ sở dữ liệu (Room Database Schema)

Dựa trên thiết kế UI, hệ thống quản lý theo phân cấp: Class (Lớp học) -> Exam (Bài thi) -> Scanned Result (Kết quả quét của từng sinh viên).

2.1 Các Entities (Bảng CSDL)

Bảng 1: ClassEntity (Lớp học)

@Entity(tableName = "classes")
data class ClassEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    @ColumnInfo(name = "name") val name: String,             // VD: "Computer Science 101"
    @ColumnInfo(name = "created_at") val createdAt: Long = System.currentTimeMillis()
)


Bảng 2: ExamEntity (Bài kiểm tra)
Chứa thông tin cấu hình của form OMR (số câu, ID length) để truyền xuống C++ config.

@Entity(
    tableName = "exams",
    foreignKeys = [ForeignKey(
        entity = ClassEntity::class,
        parentColumns = arrayOf("id"),
        childColumns = arrayOf("class_id"),
        onDelete = ForeignKey.CASCADE // Xóa lớp thì xóa luôn bài thi
    )],
    indices = [Index(value = ["class_id"])]
)
data class ExamEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    @ColumnInfo(name = "class_id") val classId: Int,
    @ColumnInfo(name = "name") val name: String,             // VD: "Midterm Exam"
    @ColumnInfo(name = "total_questions") val totalQuestions: Int, // VD: 40
    @ColumnInfo(name = "answer_keys_json") val answerKeysJson: String, // Chuỗi JSON lưu đáp án của các mã đề
    @ColumnInfo(name = "date_created") val dateCreated: Long = System.currentTimeMillis()
)


Bảng 3: ScannedResultEntity (Kết quả quét)
Lưu kết quả chi tiết của từng sinh viên. Nếu quét lại cùng student_id và exam_id, ứng dụng sẽ ghi đè (Replace).

@Entity(
    tableName = "scanned_results",
    foreignKeys = [ForeignKey(
        entity = ExamEntity::class,
        parentColumns = arrayOf("id"),
        childColumns = arrayOf("exam_id"),
        onDelete = ForeignKey.CASCADE
    )],
    indices = [
        Index(value = ["exam_id"]),
        Index(value = ["exam_id", "student_id"], unique = true) // Ngăn trùng lặp sinh viên trong 1 bài thi
    ]
)
data class ScannedResultEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    @ColumnInfo(name = "exam_id") val examId: Int,
    @ColumnInfo(name = "student_id") val studentId: String,   // VD: "2022001"
    @ColumnInfo(name = "exam_code") val examCode: String,     // VD: "101"
    @ColumnInfo(name = "raw_score") val rawScore: Int,        // VD: 38
    @ColumnInfo(name = "max_score") val maxScore: Int,        // VD: 40
    @ColumnInfo(name = "detailed_answers") val detailedAnswers: String, // JSON mapping câu hỏi - đáp án ({"1":"A", "2":"C"...})
    @ColumnInfo(name = "is_verified") val isVerified: Boolean = true,   // Cờ đánh dấu xem C++ có cảnh báo lỗi mờ/tô đúp không
    @ColumnInfo(name = "scanned_at") val scannedAt: Long = System.currentTimeMillis()
)


2.2 Data Access Object (DAO)

Cung cấp các hàm để UI (ViewModel) truy vấn dữ liệu theo thời gian thực sử dụng Flow.

@Dao
interface OMRDao {
    // ---- THAO TÁC QUÉT BÀI ----
    // Dùng OnConflictStrategy.REPLACE: Nếu sinh viên đã được chấm, quét lại sẽ cập nhật điểm mới
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertScannedResult(result: ScannedResultEntity)

    // ---- TRUY VẤN CHO GIAO DIỆN (Dashboard & Exam Details) ----
    
    // Lấy danh sách lớp học
    @Query("SELECT * FROM classes ORDER BY created_at DESC")
    fun getAllClasses(): Flow<List<ClassEntity>>

    // Lấy danh sách bài thi kèm theo số lượng đã quét (để hiển thị "Scanned: 42/45")
    @Query("""
        SELECT e.*, COUNT(r.id) as scanned_count 
        FROM exams e 
        LEFT JOIN scanned_results r ON e.id = r.exam_id 
        WHERE e.class_id = :classId 
        GROUP BY e.id
    """)
    fun getExamsForClass(classId: Int): Flow<List<ExamWithStats>>

    // Lấy kết quả của một bài thi cụ thể (cho màn hình Exam Management)
    @Query("SELECT * FROM scanned_results WHERE exam_id = :examId ORDER BY scanned_at DESC")
    fun getResultsForExam(examId: Int): Flow<List<ScannedResultEntity>>
    
    // Thống kê bài thi (Average, Highest)
    @Query("SELECT AVG(raw_score) as avgScore, MAX(raw_score) as maxScore FROM scanned_results WHERE exam_id = :examId")
    suspend fun getExamStats(examId: Int): ExamStatsData
}


3. Tóm tắt Luồng thực thi (The Pipeline)

Khi người dùng bấm "Start Scoring" trên UI:

Giao diện tải ExamEntity (bao gồm answer_keys_json).

CameraX liên tục nạp frame vào OMRImageAnalyzer.

Background Thread chạy OpenCV/C++ (so khớp luôn đáp án hoặc chỉ trả về mảng ABCD để Kotlin so khớp).

Nếu quét thành công (có thông tin Student ID), Coroutine lập tức gọi omrDao.insertScannedResult(...).

Room DB cập nhật, Flow tự động emit dữ liệu mới làm UI hiển thị Toast xanh lá "Đã chấm: 38/40" và cập nhật bộ đếm ở góc phải màn hình (Last Scored).