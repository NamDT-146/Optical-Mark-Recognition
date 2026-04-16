from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor

class AdvancedOMR:
    def __init__(self, filename="professional_omr.pdf"):
        self.c = canvas.Canvas(filename, pagesize=A4)
        self.width, self.height = A4
        # Màu sắc chủ đạo (Màu hồng đậm giống mẫu bạn gửi)
        self.primary_color = HexColor("#D81B60") 
        self.black = HexColor("#000000")

    def draw_anchors(self):
        """Vẽ điểm neo để OpenCV căn chỉnh"""
        self.c.setFillColor(self.primary_color)
        size = 8 * mm
        self.c.rect(5*mm, 5*mm, size, size, fill=1, stroke=0)
        self.c.rect(5*mm, self.height - 13*mm, size, size, fill=1, stroke=0)
        self.c.rect(self.width - 13*mm, self.height - 13*mm, size, size, fill=1, stroke=0)
        self.c.rect(self.width - 13*mm, 5*mm, size, size, fill=1, stroke=0)

    def draw_header(self):
        """Vẽ phần ghi chú đầu trang (Head Note)"""
        self.c.setStrokeColor(self.primary_color)
        self.c.setLineWidth(0.5)
        # Khung bao header
        self.c.rect(15*mm, self.height - 35*mm, self.width - 30*mm, 20*mm)
        
        self.c.setFillColor(self.primary_color)
        self.c.setFont("Helvetica-Bold", 12)
        self.c.drawCentredString(self.width/2, self.height - 23*mm, "OMR ANSWER SHEET - MOCK TEST 2026")
        
        self.c.setFillColor(self.black)
        self.c.setFont("Helvetica", 8)
        self.c.drawCentredString(self.width/2, self.height - 30*mm, 
            "Use Blue/Black Ball Pen Only. Darken the circle completely inside the box.")

    def draw_info_box(self, x, y):
        """Khu vực điền tên và thông tin (Bounding Box)"""
        self.c.setStrokeColor(self.primary_color)
        # Vẽ khung lớn cho info
        self.c.rect(x, y, 110*mm, 25*mm)
        
        # Tiêu đề box
        self.c.setFillColor(self.primary_color)
        self.c.rect(x, y + 20*mm, 110*mm, 5*mm, fill=1)
        self.c.setFillColor(HexColor("#FFFFFF"))
        self.c.setFont("Helvetica-Bold", 9)
        self.c.drawString(x + 2*mm, y + 21.5*mm, "1. CANDIDATE NAME (IN BLOCK LETTERS)")
        
        # Vẽ các ô vuông cho từng chữ cái
        self.c.setStrokeColor(HexColor("#CCCCCC"))
        for i in range(20):
            self.c.rect(x + 2*mm + (i * 5.3*mm), y + 5*mm, 5*mm, 10*mm)

    def draw_student_id_box(self, x, y, digits=10):
        """Khu vực tô mã sinh viên (ID Box)"""
        box_w = (digits * 6) * mm
        self.c.setStrokeColor(self.primary_color)
        self.c.rect(x, y, box_w, 65*mm)
        
        # Header ID
        self.c.setFillColor(self.primary_color)
        self.c.rect(x, y + 60*mm, box_w, 5*mm, fill=1)
        self.c.setFillColor(HexColor("#FFFFFF"))
        self.c.drawString(x + 2*mm, y + 61.5*mm, "2. ROLL NUMBER / ID")
        
        # Bubble grid
        self.c.setFillColor(self.black)
        for i in range(digits):
            for j in range(10):
                cx = x + 3*mm + (i * 6*mm)
                cy = y + 55*mm - (j * 5.5*mm)
                self.c.setStrokeColor(HexColor("#AAAAAA"))
                self.c.circle(cx, cy, 2.2*mm, stroke=1, fill=0)
                self.c.setFont("Helvetica", 6)
                self.c.drawCentredString(cx, cy - 1*mm, str(j))

    def draw_question_section(self, x, y, start_q, num_qs):
        """Vẽ khung câu hỏi với shading (màu nền nhạt) xen kẽ"""
        col_w = 45*mm
        self.c.setStrokeColor(self.primary_color)
        delta = 2*mm
        self.c.rect(x, y, col_w, 130*mm + delta)
        
        # Header câu hỏi
        # self.c.setFillColor(self.primary_color)
        # self.c.rect(x, y + 125*mm, col_w, 5*mm, fill=1)
        # self.c.setFillColor(HexColor("#FFFFFF"))
        # self.c.drawCentredString(x + col_w/2, y + 126.5*mm, f"Questions {start_q}-{start_q + num_qs - 1}")

        for i in range(num_qs):
            cur_y = y + 125*mm - (i * 5*mm)
            
            # Shading xen kẽ để dễ nhìn (như mẫu)
            if i % 5 < 5 and i % 10 >= 5: # Tạo các block màu hồng nhạt
                self.c.setFillColor(HexColor("#FCE4EC"))
                self.c.rect(x + 0.5*mm, cur_y - 2*mm, col_w - 1*mm, 5*mm, fill=1, stroke=0)
            
            self.c.setFillColor(self.black)
            self.c.setFont("Helvetica-Bold", 7)
            self.c.drawString(x + 2*mm, cur_y - 1*mm, f"{start_q + i:02d}")
            
            # Bubbles A, B, C, D
            for j, label in enumerate(['a', 'b', 'c', 'd']):
                bx = x + 12*mm + (j * 8*mm)
                self.c.setStrokeColor(self.primary_color)
                self.c.circle(bx, cur_y, 2*mm, stroke=1, fill=0)
                self.c.setFont("Helvetica", 6)
                self.c.drawCentredString(bx, cur_y - 0.8*mm, label)

    def draw_footer(self):
        """Bảng tổng kết cuối trang (Foot Note)"""
        y = 15*mm
        self.c.setStrokeColor(self.primary_color)
        self.c.rect(15*mm, y, self.width - 30*mm, 10*mm)
        self.c.line(55*mm, y, 55*mm, y + 10*mm)
        self.c.line(100*mm, y, 100*mm, y + 10*mm)
        
        self.c.setFillColor(self.primary_color)
        self.c.setFont("Helvetica-Bold", 8)
        self.c.drawString(18*mm, y + 4*mm, "TOTAL ATTEMPT:")
        self.c.drawString(58*mm, y + 4*mm, "CORRECT:")
        self.c.drawString(103*mm, y + 4*mm, "SCORE:")

    def build(self):
        self.draw_anchors()
        self.draw_header()
        self.draw_info_box(15*mm, 215*mm)
        self.draw_student_id_box(130*mm, 175*mm, digits=10)
        
        # Vẽ 4 cột câu hỏi (100 câu)
        for col in range(4):
            self.draw_question_section(15*mm + (col * 46*mm), 35*mm, start_q=(col*25)+1, num_qs=25)
            
        self.draw_footer()
        self.c.save()

# Chạy tạo file
omr = AdvancedOMR("test/templates/Professional_OMR_2.pdf")
omr.build()
print("Đã tạo file OMR chuyên nghiệp: Professional_OMR_2.pdf")