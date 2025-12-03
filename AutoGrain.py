import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
import random

class GrainAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grain Size & Hall-Petch Analyzer")
        self.root.geometry("1200x800")
        
        # Modern Color Palette
        self.colors = {
            "bg": "#F9F9F9",         # Lightest Gray Background
            "panel_bg": "#FFFFFF",   # White Panel
            "primary": "#1A73E8",    # Google Blue
            "success": "#34A853",    # Google Green
            "warning": "#FBBC04",    # Google Yellow/Orange
            "text": "#202124",       # Dark Gray Text
            "subtext": "#5F6368",    # Light Gray Text
            "border": "#DADCE0"      # Border Gray
        }
        
        self.root.configure(bg=self.colors["bg"])

        # --- Variables ---
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.tk_img = None
        self.pixel_scale_var = tk.DoubleVar(value=1.0)
        self.material_var = tk.StringVar(value="Steel (Low Carbon)")
        self.results_text = tk.StringVar(value="Load a micrograph to begin analysis.")
        
        # Scaling State Variables
        self.setting_scale = False
        self.scale_points = [] # Stores (x, y) tuples
        
        # Hall-Petch Constants: (Sigma_0 [MPa], k [MPa * mm^0.5])
        # updated based on standard material science texts (e.g., Dieter, Courtney)
        self.materials_db = {
            "Steel (Low Carbon)":        {"s0": 70.0,  "k": 23.0}, # Typical mild steel
            "Aluminum (1100-O Pure)":    {"s0": 15.0,  "k": 2.2},  # Pure Al is very soft, low k
            "Titanium (CP Grade 2)":     {"s0": 170.0, "k": 12.0}, # HCP metals have significant k
            "Inconel 718 (Sol. Ann.)":   {"s0": 350.0, "k": 24.0}, # Solution treated state only
            "Brass (70/30 Cartridge)":   {"s0": 70.0,  "k": 12.0}  # Added for variety
        }

        self._setup_styles()
        self._setup_ui()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam') 
        
        # Base Font
        base_font = ("Segoe UI", 10)
        header_font = ("Segoe UI", 12, "bold")
        
        # Configure Frames
        style.configure("Card.TFrame", background=self.colors["panel_bg"], relief="flat")
        
        # Configure Labels
        style.configure("TLabel", background=self.colors["panel_bg"], foreground=self.colors["text"], font=base_font)
        style.configure("Header.TLabel", font=header_font, foreground=self.colors["primary"])
        style.configure("Sub.TLabel", foreground=self.colors["subtext"], font=("Segoe UI", 9))
        
        # Configure Buttons
        style.configure("Primary.TButton", 
                        font=("Segoe UI", 10, "bold"), 
                        background=self.colors["primary"], 
                        foreground="white", 
                        borderwidth=0, 
                        focuscolor="none")
        style.map("Primary.TButton", background=[('active', '#155db5')]) 
        
        style.configure("Success.TButton", 
                        font=("Segoe UI", 10, "bold"), 
                        background=self.colors["success"], 
                        foreground="white", 
                        borderwidth=0,
                        focuscolor="none")
        style.map("Success.TButton", background=[('active', '#2d8e47')]) 

        style.configure("Warning.TButton", 
                        font=("Segoe UI", 9, "bold"), 
                        background=self.colors["warning"], 
                        foreground="white", 
                        borderwidth=0,
                        focuscolor="none")
        style.map("Warning.TButton", background=[('active', '#e5ac04')])

        # Entry
        style.configure("TEntry", fieldbackground="white", borderwidth=1)

    def _setup_ui(self):
        # Main Layout: Sidebar (Controls) + Main Area (Canvas)
        
        # --- Sidebar ---
        sidebar = ttk.Frame(self.root, width=320, style="Card.TFrame", padding=20)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False) # Force width

        # App Title
        ttk.Label(sidebar, text="Grain Analyzer", font=("Segoe UI", 18, "bold"), foreground=self.colors["text"]).pack(anchor="w", pady=(0, 5))
        ttk.Label(sidebar, text="v1.1 • Hall-Petch Method", style="Sub.TLabel").pack(anchor="w", pady=(0, 30))

        # 1. Load Section
        self._create_section_header(sidebar, "1. Input Data")
        
        btn_load = ttk.Button(sidebar, text="📂  Upload Micrograph", style="Primary.TButton", command=self.load_image)
        btn_load.pack(fill=tk.X, pady=(10, 15), ipady=5)

        # Scale Section
        ttk.Label(sidebar, text="Scale (Pixels per µm):").pack(anchor="w", pady=(5, 0))
        
        scale_frame = ttk.Frame(sidebar, style="Card.TFrame")
        scale_frame.pack(fill=tk.X, pady=(5, 0))
        
        scale_entry = ttk.Entry(scale_frame, textvariable=self.pixel_scale_var, width=15)
        scale_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_measure = ttk.Button(scale_frame, text="Measure on Image", style="Warning.TButton", command=self.activate_scale_tool)
        self.btn_measure.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(sidebar, text="Manual: Enter value above.\nAuto: Click 'Measure', then click 2 points.", style="Sub.TLabel").pack(anchor="w", pady=(2,0))

        # Material
        ttk.Label(sidebar, text="Material Selection:").pack(anchor="w", pady=(20, 0))
        mat_options = list(self.materials_db.keys())
        self.material_dropdown = ttk.Combobox(sidebar, textvariable=self.material_var, values=mat_options, state="readonly")
        self.material_dropdown.pack(fill=tk.X, pady=5)
        
        # 2. Action Section
        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=25)
        
        btn_analyze = ttk.Button(sidebar, text="⚡  Run Analysis", style="Success.TButton", command=self.analyze_grains)
        btn_analyze.pack(fill=tk.X, ipady=5)

        # 3. Results Section
        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=25)
        self._create_section_header(sidebar, "Results")
        
        self.result_label = tk.Label(sidebar, textvariable=self.results_text, 
                                     bg="#F1F3F4", fg=self.colors["text"],
                                     justify=tk.LEFT, anchor="nw",
                                     font=("Consolas", 10), 
                                     padx=10, pady=10, relief=tk.FLAT)
        self.result_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0))


        # --- Main Canvas Area ---
        canvas_frame = tk.Frame(self.root, bg=self.colors["bg"])
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Shadow/Card effect for canvas container
        self.canvas_container = tk.Frame(canvas_frame, bg="white", bd=1, relief=tk.SOLID)
        self.canvas_container.config(highlightbackground=self.colors["border"], highlightthickness=1)
        self.canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(self.canvas_container, bg="#E8EAED", highlightthickness=0, cursor="crosshair")
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Bind Click Event
        self.image_canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Initial Placeholder
        self.canvas_text = self.image_canvas.create_text(
            400, 350, 
            text="No Image Loaded", 
            fill="#9AA0A6", 
            font=("Segoe UI", 24, "bold")
        )

    def _create_section_header(self, parent, text):
        lbl = ttk.Label(parent, text=text, style="Header.TLabel")
        lbl.pack(anchor="w")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if not file_path:
            return

        self.image_path = file_path
        
        # Read image with OpenCV
        img = cv2.imread(self.image_path)
        if img is None:
            messagebox.showerror("Error", "Could not read image file.")
            return

        self.original_image = img
        self.processed_image = img.copy()
        
        self.display_image(self.original_image)
        self.results_text.set("Image Loaded.\n1. Set Scale (Manual or Measure).\n2. Select Material.\n3. Click Run Analysis.")

    def display_image(self, cv_image):
        """Resizes and displays CV2 image on Tkinter Canvas"""
        if cv_image is None:
            return

        self.root.update()
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Handle initial load size 0
        if canvas_width < 10: canvas_width = 800
        if canvas_height < 10: canvas_height = 600

        # Calculate aspect ratio
        h, w = cv_image.shape[:2]
        self.current_scale_ratio = min(canvas_width/w, canvas_height/h)
        new_w, new_h = int(w*self.current_scale_ratio), int(h*self.current_scale_ratio)

        resized = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(resized_rgb)
        
        self.tk_img = ImageTk.PhotoImage(pil_img)

        self.image_canvas.delete("all")
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        
        # Store offset to translate canvas clicks to image coordinates later
        self.img_offset_x = x_center - (new_w // 2)
        self.img_offset_y = y_center - (new_h // 2)
        
        self.image_canvas.create_image(x_center, y_center, image=self.tk_img, anchor=tk.CENTER)

    # --- Scaling Tool Logic ---
    def activate_scale_tool(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return
        
        self.setting_scale = True
        self.scale_points = []
        self.results_text.set("MEASURE MODE ACTIVE:\nClick on the start of the scale bar,\nthen click on the end.")
        self.btn_measure.configure(text="Click 2 Points...", state="disabled")

    def on_canvas_click(self, event):
        if not self.setting_scale:
            return

        # Draw visual feedback
        x, y = event.x, event.y
        self.scale_points.append((x, y))
        
        # Draw small circle at click
        r = 3
        self.image_canvas.create_oval(x-r, y-r, x+r, y+r, fill="#FBBC04", outline="black")

        if len(self.scale_points) == 2:
            self._finalize_scale_measurement()

    def _finalize_scale_measurement(self):
        p1 = self.scale_points[0]
        p2 = self.scale_points[1]
        
        # Draw line connecting them
        self.image_canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#FBBC04", width=2, arrow=tk.BOTH)
        
        # Euclidean distance in Canvas Pixels
        dist_canvas_px = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        
        # Convert back to Original Image Pixels
        # Dist_Original = Dist_Canvas / Scale_Ratio
        try:
            dist_original_px = dist_canvas_px / self.current_scale_ratio
        except ZeroDivisionError:
             dist_original_px = dist_canvas_px # Should not happen

        # Ask user for real world length
        length_microns = simpledialog.askfloat("Input Length", 
                                             f"Pixel Distance: {dist_original_px:.1f} px\n\n"
                                             "Enter known length of this bar (in microns):",
                                             minvalue=0.1, maxvalue=10000)
        
        if length_microns:
            px_per_micron = dist_original_px / length_microns
            self.pixel_scale_var.set(round(px_per_micron, 4))
            self.results_text.set(f"Scale Set!\n{px_per_micron:.2f} pixels = 1 micron.")
        else:
            self.results_text.set("Measurement Cancelled.")
            # Clear drawing if cancelled
            self.display_image(self.original_image)

        # Reset State
        self.setting_scale = False
        self.btn_measure.configure(text="Measure on Image", state="normal")

    # --- Analysis Logic ---
    def analyze_grains(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            scale_factor = self.pixel_scale_var.get()
            if scale_factor <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid Pixel Scale.\nPlease enter a number > 0.")
            return

        # 1. Image Processing Pipeline
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((2,2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        edges = cv2.Canny(blurred, 50, 150)
        combined_edges = cv2.bitwise_or(edges, opening)

        # 2. ASTM Circular Intercept Method
        h, w = combined_edges.shape
        visualization = self.original_image.copy()
        
        num_circles = 5
        min_dim = min(h, w)
        radius = int(min_dim * 0.35)
        
        total_intercepts = 0
        total_circumference_px = 0

        for i in range(num_circles):
            center_x = w // 2 + random.randint(-int(w*0.1), int(w*0.1))
            center_y = h // 2 + random.randint(-int(h*0.1), int(h*0.1))
            
            if center_x - radius < 0 or center_x + radius > w or center_y - radius < 0 or center_y + radius > h:
                center_x, center_y = w // 2, h // 2

            circle_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(circle_mask, (center_x, center_y), radius, 255, 1)
            
            intersections = cv2.bitwise_and(circle_mask, combined_edges)
            contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            n_intercepts = len(contours)
            total_intercepts += n_intercepts
            total_circumference_px += (2 * math.pi * radius)

            cv2.circle(visualization, (center_x, center_y), radius, (0, 0, 255), 2)
            for cnt in contours:
                x,y,w_c,h_c = cv2.boundingRect(cnt)
                cv2.circle(visualization, (x+w_c//2, y+h_c//2), 3, (255, 255, 0), -1)

        # 3. Calculations (Triple Checked)
        if total_intercepts == 0:
            self.results_text.set("Analysis Failed: No boundaries found.\nTry a higher contrast image.")
            self.display_image(combined_edges)
            return

        # Mean Lineal Intercept (L) in pixels
        avg_intercept_px = total_circumference_px / total_intercepts
        
        # Convert to microns (L_um)
        # Check: px / (px/um) = um. Correct.
        avg_grain_intercept_um = avg_intercept_px / scale_factor
        
        # Convert to mm for Hall-Petch (d_mm)
        # Check: um / 1000 = mm. Correct.
        d_mm = avg_grain_intercept_um / 1000.0

        # Hall-Petch Relation
        # Formula: sigma_y = sigma_0 + k * d^(-1/2)
        # Units: MPa = MPa + (MPa * mm^0.5) * (mm)^-0.5
        # Units Check: mm^0.5 * mm^-0.5 = 1 (dimensionless). Result is MPa. Correct.
        
        mat_data = self.materials_db[self.material_var.get()]
        s0 = mat_data["s0"]
        k = mat_data["k"]
        
        try:
            yield_strength = s0 + (k * (d_mm ** -0.5))
        except ZeroDivisionError:
            yield_strength = 0

        # 4. Generate Report
        res_str = (
            f"MATERIAL: {self.material_var.get()}\n"
            f"----------------------------------------\n"
            f"Intercepts Counted : {total_intercepts}\n"
            f"Mean Lineal Intercept: {avg_grain_intercept_um:.2f} µm\n"
            f"ASTM Grain Number (G): {self.calculate_astm(avg_grain_intercept_um):.2f}\n\n"
            f"MECHANICAL PROPERTIES (EST.)\n"
            f"----------------------------------------\n"
            f"Formula: σy = σ₀ + k·d⁻¹/²\n"
            f"Grain Diam (d)     : {d_mm:.4f} mm\n"
            f"Friction Stress σ₀ : {s0} MPa\n"
            f"Locking Param. k   : {k} MPa·√mm\n"
            f"Yield Strength σy  : {int(yield_strength)} MPa"
        )
        
        self.results_text.set(res_str)
        self.display_image(visualization)

    def calculate_astm(self, mean_intercept_um):
        """
        Calculates ASTM E112 Grain Size Number (G).
        Formula: G = -6.643856 * log10(L_mm) - 3.288
        Where L_mm is the mean lineal intercept length in mm.
        """
        if mean_intercept_um <= 0: return 0
        l_mm = mean_intercept_um / 1000.0
        return -6.643856 * math.log10(l_mm) - 3.288

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = GrainAnalyzerApp(root)
    root.mainloop()
