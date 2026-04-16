from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

class DynamicOMRTemplate:
    def __init__(self, filename="omr_form.pdf"):
        self.c = canvas.Canvas(filename, pagesize=A4)
        self.width, self.height = A4

    def draw_anchor_points(self):
        """Draw black squares at corners for Perspective Transform (Required on EVERY page)"""
        size = 10 * mm
        self.c.rect(10*mm, 10*mm, size, size, fill=1)
        self.c.rect(10*mm, self.height - 20*mm, size, size, fill=1)
        self.c.rect(self.width - 20*mm, self.height - 20*mm, size, size, fill=1)
        self.c.rect(self.width - 20*mm, 10*mm, size, size, fill=1)

    def draw_student_info(self, x, y):
        """Draws the Name, Class, and Exam Code blocks on the First Page"""
        self.c.setFont("Helvetica-Bold", 14)
        self.c.drawString(x, y, "AUTOMATED OMR ANSWER SHEET")
        
        self.c.setFont("Helvetica", 11)
        self.c.drawString(x, y - 15*mm, "Full Name: ____________________________________")
        self.c.drawString(x, y - 30*mm, "Class: _________________    Date: _____________")
        
        # Exam Code Bubble Grid (3 digits)
        self.c.setFont("Helvetica-Bold", 10)
        self.c.drawString(x, y - 50*mm, "EXAM CODE")
        self.c.setFont("Helvetica", 7)
        for i in range(3):
            for j in range(10):
                cur_x = x + (i * 6 * mm)
                cur_y = y - 55*mm - (j * 5 * mm)
                self.c.circle(cur_x + 2.5*mm, cur_y - 2.5*mm, 2*mm, stroke=1, fill=0)
                if i == 0:
                    self.c.drawString(cur_x - 3*mm, cur_y - 3.5*mm, str(j))

    def draw_student_id(self, x, y, num_digits=8):
        """Draws the Student ID grid"""
        self.c.setFont("Helvetica-Bold", 11)
        self.c.drawString(x, y + 5*mm, "STUDENT ID NUMBER")
        
        cell_size = 5 * mm
        for i in range(num_digits):
            for j in range(10): # Digits 0-9
                cur_x = x + (i * cell_size)
                cur_y = y - (j * cell_size)
                self.c.circle(cur_x + 2.5*mm, cur_y - 2.5*mm, 2*mm, stroke=1, fill=0)
                if i == 0: # Draw numbers on the left margin
                    self.c.setFont("Helvetica", 7)
                    self.c.drawString(x - 4*mm, cur_y - 3.5*mm, str(j))

    def draw_question_grid(self, start_q, count, num_opts, x_start, y_start, rows_per_col):
        """Draws a grid of questions automatically flowing into columns"""
        letters = ["A", "B", "C", "D", "E"]
        line_height = 7.5 * mm   # Vertical spacing between questions
        col_width = 42 * mm      # Horizontal spacing between columns
        opt_width = 6.5 * mm     # Spacing between A, B, C, D bubbles
        
        for i in range(count):
            q_num = start_q + i
            col = i // rows_per_col   # Determine which column (0, 1, 2, 3)
            row = i % rows_per_col    # Determine which row in that column
            
            cur_x = x_start + (col * col_width)
            cur_y = y_start - (row * line_height)
            
            # Draw Question Number
            self.c.setFont("Helvetica-Bold", 9)
            self.c.drawString(cur_x, cur_y - 1*mm, f"{q_num:02d}.")
            
            # Draw Bubbles
            self.c.setFont("Helvetica", 7)
            for j in range(num_opts):
                opt_x = cur_x + 9*mm + (j * opt_width)
                self.c.circle(opt_x, cur_y, 2.2*mm, stroke=1, fill=0)
                self.c.drawCentredString(opt_x, cur_y - 1*mm, letters[j])

    def generate(self, total_qs, num_opts=4, id_digits=8):
        """The Main Controller: Handles logic for splitting questions across pages"""
        current_q = 1
        page = 1
        remaining_qs = total_qs
        
        while remaining_qs > 0:
            if page > 1:
                self.c.showPage() # Creates a new PDF page
            
            # Every page needs anchor points for OpenCV
            self.draw_anchor_points()
            
            if page == 1:
                # Page 1 Setup: Has Header, max 80 questions
                self.draw_student_info(25*mm, 275*mm)
                self.draw_student_id(140*mm, 260*mm, num_digits=id_digits)
                
                qs_this_page = min(remaining_qs, 80) # 4 columns * 20 rows
                self.draw_question_grid(
                    start_q=current_q, count=qs_this_page, num_opts=num_opts, 
                    x_start=20*mm, y_start=190*mm, rows_per_col=20
                )
            else:
                # Page 2+ Setup: No Header, max 120 questions
                qs_this_page = min(remaining_qs, 120) # 4 columns * 30 rows
                self.draw_question_grid(
                    start_q=current_q, count=qs_this_page, num_opts=num_opts, 
                    x_start=20*mm, y_start=270*mm, rows_per_col=30
                )
            
            # Update counters
            current_q += qs_this_page
            remaining_qs -= qs_this_page
            page += 1
            
        self.c.save()
        print(f"Success: Generated OMR with {total_qs} questions across {page-1} page(s).")

# --- Execution Tests ---
if __name__ == "__main__":
    # Test 1: Short quiz (Fits on Page 1)
    form1 = DynamicOMRTemplate("test/templates/omr_45qs.pdf")
    form1.generate(total_qs=45, num_opts=4)

    # Test 2: Final Exam (Needs 2 pages: 80 on P1, 120 on P2)
    form2 = DynamicOMRTemplate("test/templates/omr_200qs.pdf")
    form2.generate(total_qs=200, num_opts=4, id_digits=8)