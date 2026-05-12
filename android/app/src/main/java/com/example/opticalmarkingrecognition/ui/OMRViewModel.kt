package com.example.opticalmarkingrecognition.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.opticalmarkingrecognition.database.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class OMRViewModel(application: Application) : AndroidViewModel(application) {

    private val dao: OMRDao = OMRDatabase.getDatabase(application).omrDao()

    // ── Navigation state ──
    private val _selectedClassId = MutableStateFlow<Int?>(null)
    val selectedClassId: StateFlow<Int?> = _selectedClassId.asStateFlow()

    private val _selectedExamId = MutableStateFlow<Int?>(null)
    val selectedExamId: StateFlow<Int?> = _selectedExamId.asStateFlow()

    // ── Data flows ──
    val allClasses: StateFlow<List<ClassEntity>> = dao.getAllClasses()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val examsForSelectedClass: StateFlow<List<ExamWithStats>> =
        _selectedClassId.flatMapLatest { classId ->
            if (classId != null) dao.getExamsForClass(classId)
            else flowOf(emptyList())
        }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val resultsForSelectedExam: StateFlow<List<ScannedResultEntity>> =
        _selectedExamId.flatMapLatest { examId ->
            if (examId != null) dao.getResultsForExam(examId)
            else flowOf(emptyList())
        }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    // ── Exam stats ──
    private val _examStats = MutableStateFlow(ExamStatsData(null, null))
    val examStats: StateFlow<ExamStatsData> = _examStats.asStateFlow()

    private val _selectedExam = MutableStateFlow<ExamEntity?>(null)
    val selectedExam: StateFlow<ExamEntity?> = _selectedExam.asStateFlow()

    // ── Actions ──
    fun selectClass(classId: Int) {
        _selectedClassId.value = classId
        _selectedExamId.value = null
    }

    fun selectExam(examId: Int) {
        _selectedExamId.value = examId
        viewModelScope.launch(Dispatchers.IO) {
            _examStats.value = dao.getExamStats(examId)
            _selectedExam.value = dao.getExamById(examId)
        }
    }

    fun clearClassSelection() {
        _selectedClassId.value = null
        _selectedExamId.value = null
    }

    fun clearExamSelection() {
        _selectedExamId.value = null
    }

    fun addClass(name: String) {
        viewModelScope.launch(Dispatchers.IO) {
            dao.insertClass(ClassEntity(name = name))
        }
    }

    fun addExam(classId: Int, name: String, totalQuestions: Int, answerKeysJson: String = "{}") {
        viewModelScope.launch(Dispatchers.IO) {
            dao.insertExam(
                ExamEntity(
                    classId = classId,
                    name = name,
                    totalQuestions = totalQuestions,
                    answerKeysJson = answerKeysJson
                )
            )
        }
    }

    fun deleteClass(classEntity: ClassEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            dao.deleteClass(classEntity)
        }
    }

    fun deleteExam(examEntity: ExamEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            dao.deleteExam(examEntity)
        }
    }

    fun insertScannedResult(result: ScannedResultEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            dao.insertScannedResult(result)
            // Refresh stats
            _selectedExamId.value?.let { examId ->
                _examStats.value = dao.getExamStats(examId)
            }
        }
    }
}
