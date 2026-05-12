package com.example.opticalmarkingrecognition.database

import android.content.Context
import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "classes")
data class ClassEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    @ColumnInfo(name = "name") val name: String,             // VD: "Computer Science 101"
    @ColumnInfo(name = "created_at") val createdAt: Long = System.currentTimeMillis()
)

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

data class ExamWithStats(
    @Embedded val exam: ExamEntity,
    @ColumnInfo(name = "scanned_count") val scannedCount: Int
)

data class ExamStatsData(
    val avgScore: Double?,
    val maxScore: Int?
)

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

    @Insert
    suspend fun insertClass(classEntity: ClassEntity): Long

    @Insert
    suspend fun insertExam(examEntity: ExamEntity): Long

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

    @Delete
    suspend fun deleteClass(classEntity: ClassEntity): Int // <-- Return Int for Delete

    @Delete
    suspend fun deleteExam(examEntity: ExamEntity): Int

    @Delete
    suspend fun deleteScannedResult(result: ScannedResultEntity): Int

    @Query("SELECT * FROM exams WHERE id = :examId")
    suspend fun getExamById(examId: Int): ExamEntity?

    @Query("SELECT COUNT(*) FROM scanned_results WHERE exam_id = :examId")
    suspend fun getScannedCountForExam(examId: Int): Int
}

@Database(entities = [ClassEntity::class, ExamEntity::class, ScannedResultEntity::class], version = 1, exportSchema = false)
abstract class OMRDatabase : RoomDatabase() {
    abstract fun omrDao(): OMRDao

    companion object {
        @Volatile
        private var INSTANCE: OMRDatabase? = null

        fun getDatabase(context: Context): OMRDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    OMRDatabase::class.java,
                    "omr_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
