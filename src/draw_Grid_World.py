from tkinter import Tk, Canvas, Frame, BOTH


BORDER_WIDTH = 10
BOX_WIDTH = BOX_HEIGHT = 90
GRID_COUNT = 4
class Render_UI(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()


    def draw_box(self, canvas, x1, y1, color):
        # offset
        x1+=BORDER_WIDTH
        y1+=BORDER_WIDTH
        
        canvas.create_rectangle(x1, y1, x1+BOX_WIDTH, y1+BOX_HEIGHT,
            outline=color, fill=color)
    

    def initUI(self):
        self.master.title("Grid World")
        self.pack(fill=BOTH, expand=1)

        canvas = Canvas(self)
        for i in range(GRID_COUNT):
            for j in range(GRID_COUNT):
                if i==0 and j==0:
                    self.draw_box(canvas, i*(BOX_WIDTH+BORDER_WIDTH), 
                        j*(BOX_HEIGHT+BORDER_WIDTH), "green")
                elif i==GRID_COUNT-1 and j==GRID_COUNT-1:
                    self.draw_box(canvas, i*(BOX_WIDTH+BORDER_WIDTH), 
                        j*(BOX_HEIGHT+BORDER_WIDTH), "blue")
                else:
                    self.draw_box(canvas, i*(BOX_WIDTH+BORDER_WIDTH), 
                        j*(BOX_HEIGHT+BORDER_WIDTH), "purple")
                
        canvas.pack(fill=BOTH, expand=1)


def main():
    root = Tk()
    ex = Render_UI()
    root.geometry(f"{GRID_COUNT}10x{GRID_COUNT}10+100+100")
    root.mainloop()


if __name__ == '__main__':
    main()