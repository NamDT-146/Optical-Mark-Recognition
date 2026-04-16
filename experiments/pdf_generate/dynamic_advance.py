from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
import json

# Cấu hình màu sắc (Sử dụng Đen/Xám để tối ưu cho máy in trắng đen)
BLACK = HexColor("#000000")
GRAY_DARK = HexColor("#333333")
GRAY_MEDIUM = HexColor("#666666")
GRAY_LIGHT = HexColor("#EEEEEE")
WHITE = HexColor("#FFFFFF")

class OMRRenderer:
    """Lớp hỗ trợ vẽ các thành phần cơ bản của OMR"""
    def __init__(self, canvas_obj):
        self.c = canvas_obj
        self.width, self.height = A4

    def draw_anchors(self):
        self.c.setFillColor(BLACK)
        size = 8 * mm
        self.c.rect(5*mm, 5*mm, size, size, fill=1, stroke=0)
        self.c.rect(5*mm, self.height - 13*mm, size, size, fill=1, stroke=0)
        self.c.rect(self.width - 13*mm, self.height - 13*mm, size, size, fill=1, stroke=0)
        self.c.rect(self.width - 13*mm, 5*mm, size, size, fill=1, stroke=0)

    def draw_boxed_header(self, title):
        self.c.setStrokeColor(BLACK)
        self.c.rect(15*mm, self.height - 30*mm, self.width - 30*mm, 15*mm)
        self.c.setFillColor(BLACK)
        self.c.setFont("Helvetica-Bold", 12)
        self.c.drawCentredString(self.width/2, self.height - 21*mm, title)
        self.c.setFillColor(BLACK)
        self.c.setFont("Helvetica", 7)
        self.c.drawCentredString(self.width/2, self.height - 27*mm, "Use Blue/Black Ball Pen. Darken the circle completely.")

    def draw_student_info_box(self, x, y):
        # Name Box
        self.c.setStrokeColor(BLACK)
        self.c.rect(x, y, 110*mm, 20*mm)
        self.c.setFillColor(BLACK)
        self.c.rect(x, y + 15*mm, 110*mm, 5*mm, fill=1)
        self.c.setFillColor(WHITE)
        self.c.setFont("Helvetica-Bold", 8)
        self.c.drawString(x + 2*mm, y + 16.5*mm, "FULL NAME (IN BLOCK LETTERS)")
        # Letter boxes
        self.c.setStrokeColor(GRAY_MEDIUM)
        for i in range(20):
            self.c.rect(x + 2*mm + (i * 5.3*mm), y + 3*mm, 5*mm, 8*mm)

    def draw_bubble_grid(self, x, y, title, cols, rows=10, add_write_row=False):
        """Dùng cho Student ID hoặc Exam Code
        - add_write_row: Thêm hàng viết tay (ô trắng) ở phía trên grid bubbles
        """
        box_w = (cols * 6) * mm
        write_row_h_mm = 9 if add_write_row else 0
        box_h = (rows * 5 + 10 + write_row_h_mm) * mm
        self.c.setStrokeColor(BLACK)
        self.c.rect(x, y, box_w, box_h)
        self.c.setFillColor(BLACK)
        self.c.rect(x, y + box_h - 5*mm, box_w, 5*mm, fill=1)
        self.c.setFillColor(WHITE)
        self.c.setFont("Helvetica-Bold", 7)
        self.c.drawCentredString(x + box_w/2, y + box_h - 3.5*mm, title)
        
        # Vẽ hàng viết tay ở phía trên nếu có
        if add_write_row:
            self.c.setStrokeColor(GRAY_DARK)
            for i in range(cols):
                self.c.rect(x + (i * 6 + 0.5) * mm, y + (rows * 5 + 3) * mm, 5*mm, 5*mm)
        
        self.c.setFillColor(BLACK)
        bubble_top_y = y + (rows * 5) * mm
        for i in range(cols):
            for j in range(rows):
                cx = x + 3*mm + (i * 6*mm)
                cy = bubble_top_y - (j * 5*mm)
                self.c.setStrokeColor(GRAY_DARK)
                self.c.circle(cx, cy, 2*mm, stroke=1, fill=0)
                self.c.setFont("Helvetica", 6)
                self.c.drawCentredString(cx, cy - 0.8*mm, str(j))

    def draw_questions(self, x, y, start_q, num_qs, rows_per_col=20):
        col_w = 42 * mm
        num_cols = (num_qs + rows_per_col - 1) // rows_per_col
        
        for c in range(num_cols):
            curr_x = x + (c * 43*mm)
            qs_in_this_col = min(rows_per_col, num_qs - (c * rows_per_col))
            
            # Draw Column Box
            self.c.setStrokeColor(BLACK)
            self.c.rect(curr_x, y - (rows_per_col * 5.5*mm), col_w, (rows_per_col * 5.5*mm) + 5*mm)
            
            for r in range(qs_in_this_col):
                q_num = start_q + (c * rows_per_col) + r
                row_y = y - (r * 5.5*mm)
                
                # Shading xen kẽ (Sử dụng xám rất nhạt)
                if r % 10 >= 5:
                    self.c.setFillColor(GRAY_LIGHT)
                    self.c.rect(curr_x + 0.2*mm, row_y - 2*mm, col_w - 0.4*mm, 5.5*mm, fill=1, stroke=0)
                
                self.c.setFillColor(BLACK)
                self.c.setFont("Helvetica-Bold", 7)
                self.c.drawString(curr_x + 2*mm, row_y, f"{q_num:02d}")
                
                # Options A, B, C, D
                for j, label in enumerate(['A', 'B', 'C', 'D']):
                    bx = curr_x + 10*mm + (j * 8*mm)
                    self.c.setStrokeColor(BLACK)
                    self.c.circle(bx, row_y + 1*mm, 2.1*mm, stroke=1, fill=0)
                    self.c.setFont("Helvetica", 5)
                    self.c.drawCentredString(bx, row_y + 0.2*mm, label)

class ProfessionalGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.c = canvas.Canvas(filename, pagesize=A4)
        self.renderer = OMRRenderer(self.c)
        self.metadata = {
            "filename": filename,
            "format_version": 2,
            "reference_dpi": 254,
            "page_size_px": {
                "width": self.mm_to_px(210),
                "height": self.mm_to_px(297)
            },
            "pages": []
        }

    def mm_to_px(self, mm_val, dpi=254):
        """Chuyển đổi mm sang pixel (Dựa trên độ phân giải mục tiêu của Parser)"""
        # A4: 210mm x 297mm. Ở 254 DPI, 1mm = 10px -> 2100px x 2970px
        return round(mm_val * 10)

    def get_y_top_down(self, y_bottom_up_mm):
        """
        ReportLab dùng gốc (0,0) ở dưới-trái. OpenCV dùng ở trên-trái.
        Hàm này chuyển tọa độ y từ Bottom-Up sang Top-Down.
        """
        return 297 - y_bottom_up_mm

    def build_questions_roi_metadata(self, x_mm, y_mm, start_q, num_qs, rows_per_col):
        """
        Build metadata for question area using the exact same geometry as draw_questions.
        The exported coordinates are top-down pixel coordinates (for OpenCV parser).
        """
        col_w_mm = 42
        row_h_mm = 5.5
        col_step_mm = 43
        option_start_x_mm = 10
        option_step_x_mm = 8
        bubble_center_y_offset_mm = -1  # row_y + 1mm in bottom-up => -1mm in top-down
        bubble_radius_mm = 2.1

        roi = {
            "start_q": start_q,
            "num_qs": num_qs,
            "x": self.mm_to_px(x_mm),
            "y": self.mm_to_px(self.get_y_top_down(y_mm)),
            "col_w": self.mm_to_px(col_w_mm),
            "row_h": self.mm_to_px(row_h_mm),
            "rows_per_col": rows_per_col,
            "col_step": self.mm_to_px(col_step_mm),
            "option_start_x": self.mm_to_px(option_start_x_mm),
            "option_step_x": self.mm_to_px(option_step_x_mm),
            "bubble_center_y_offset": self.mm_to_px(bubble_center_y_offset_mm),
            "bubble_radius": self.mm_to_px(bubble_radius_mm),
            "options": ["A", "B", "C", "D"],
            "bubble_centers": {}
        }

        for q_idx in range(num_qs):
            q_num = start_q + q_idx
            c = q_idx // rows_per_col
            r = q_idx % rows_per_col

            curr_x_mm = x_mm + (c * col_step_mm)
            row_y_mm = y_mm - (r * row_h_mm)
            center_y_top_down_mm = self.get_y_top_down(row_y_mm + 1)

            option_centers = {}
            for j, label in enumerate(roi["options"]):
                center_x_mm = curr_x_mm + option_start_x_mm + (j * option_step_x_mm)
                option_centers[label] = {
                    "x": self.mm_to_px(center_x_mm),
                    "y": self.mm_to_px(center_y_top_down_mm)
                }

            roi["bubble_centers"][str(q_num)] = option_centers

        return roi

    def generate(self, total_qs, save_pdf=True):
        current_q = 1
        page_num = 1
        
        while current_q <= total_qs:
            page_info = {"page_number": page_num, "rois": {}}
            if save_pdf and page_num > 1:
                self.c.showPage()

            if save_pdf:
                self.renderer.draw_anchors()
            
            # --- TRANG 1 ---
            if page_num == 1:
                if save_pdf:
                    self.renderer.draw_boxed_header("OMR ANSWER SHEET")
                
                # Lưu tọa độ Name Box
                y_mm = 240
                page_info["rois"]["name_box"] = {
                    "x": self.mm_to_px(15), "y": self.mm_to_px(self.get_y_top_down(y_mm + 20)),
                    "w": self.mm_to_px(110), "h": self.mm_to_px(20)
                }
                if save_pdf:
                    self.renderer.draw_student_info_box(15*mm, y_mm*mm)
                
                # Lưu tọa độ Grids
                y_grid = 160
                page_info["rois"]["exam_code"] = {
                    "x": self.mm_to_px(15), "y": self.mm_to_px(self.get_y_top_down(y_grid + 25)),
                    "w": self.mm_to_px(18), "h": self.mm_to_px(25), "cols": 3, "rows": 10
                }
                if save_pdf:
                    self.renderer.draw_bubble_grid(15*mm, y_grid*mm, "EXAM CODE", cols=3, add_write_row=True)
                
                # Student ID bên phải Exam Code
                student_id_cols = 10  # Config độ dài Student ID (default 10 chữ số)
                student_id_x = 15 + 18 + 5  # Bên phải Exam Code + khoảng cách 5mm
                page_info["rois"]["student_id"] = {
                    "x": self.mm_to_px(student_id_x), "y": self.mm_to_px(self.get_y_top_down(y_grid + 25)),
                    "w": self.mm_to_px(student_id_cols * 6), "h": self.mm_to_px(31), "cols": student_id_cols, "rows": 10
                }
                if save_pdf:
                    self.renderer.draw_bubble_grid(student_id_x*mm, y_grid*mm, "STUDENT ID", cols=student_id_cols, add_write_row=True)
                
                # Lưu tọa độ Questions
                qs_to_draw = min(total_qs - current_q + 1, 80)
                y_qs = 145
                page_info["rois"]["questions"] = self.build_questions_roi_metadata(
                    x_mm=15,
                    y_mm=y_qs,
                    start_q=current_q,
                    num_qs=qs_to_draw,
                    rows_per_col=20
                )
                if save_pdf:
                    self.renderer.draw_questions(15*mm, y_qs*mm, current_q, qs_to_draw)
                current_q += qs_to_draw
            
            else:  # page_num > 1
                # Subsequent pages: full-page question layout
                qs_to_draw = min(total_qs - current_q + 1, 140)
                # Đã hạ y_qs từ 280 xuống 275 để top-box không đè vào Anchor góc trên
                y_qs = 275 
                
                page_info["rois"]["questions"] = self.build_questions_roi_metadata(
                    x_mm=15,
                    y_mm=y_qs,
                    start_q=current_q,
                    num_qs=qs_to_draw,
                    rows_per_col=35
                )
                if save_pdf:
                    self.renderer.draw_questions(15*mm, y_qs*mm, current_q, qs_to_draw, rows_per_col=35)
                current_q += qs_to_draw

            self.metadata["pages"].append(page_info)
            page_num += 1
            
        if save_pdf:
            self.c.save()
        
        # Xuất Metadata ra file JSON cùng tên
        with open(self.filename.replace(".pdf", ".json"), "w") as f:
            json.dump(self.metadata, f, indent=4)

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    # Case 1: 120 câu (80 trang đầu, 40 trang sau)
    gen = ProfessionalGenerator("test/templates/Professional_OMR_120.pdf")
    gen.generate(120)
    
    # Case 2: 200 câu (80 trang đầu, 120 trang sau)
    gen2 = ProfessionalGenerator("test/templates/Professional_OMR_200.pdf")
    gen2.generate(200)
    print("Success: Generated flexible professional OMRs.")

    gen3 = ProfessionalGenerator("test/templates/Professional_OMR_45.pdf")
    gen3.generate(45, save_pdf=False)
    print("Success: Generated OMR with 45 questions on a single page.")

    gen4 = ProfessionalGenerator("test/templates/Professional_OMR_300.pdf")
    gen4.generate(300)
    print("Success: Generated OMR with 300 questions across multiple pages.")