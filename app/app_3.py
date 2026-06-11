class FaceAuthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hệ Thống Xác Thực Khuôn Mặt")
        self.geometry("1280x720")
        self.configure(bg=C_BG)
        try:
            self.state("zoomed")
        except:
            self.attributes("-zoomed", True)

        try:
            self.vn_font_large = ImageFont.truetype("arial.ttf", 36)
            self.vn_font_medium = ImageFont.truetype("arial.ttf", 24)
            self.vn_font_small = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            self.vn_font_large = ImageFont.load_default()
            self.vn_font_medium = ImageFont.load_default()
            self.vn_font_small = ImageFont.load_default()

        self.detector = cv2.CascadeClassifier(HAAR_CASCADE_XML)
        self.recognizer = None
        self._load_models()
        self.label_map = load_label_map()
        self.face_mask = create_face_mask(TARGET_SIZE)

        self.cap = None
        self.running = False
        self.app_state = "init"
        self.mode = "auth"
        self.is_capturing = False
        self.frame_idx = 0
        self.quality_warning = ""

        self.reg_id = None
        self.reg_name = ""
        self.reg_count = 0

        self._primary_face = None
        self._last_result = None

        self.root_container = tk.Frame(self, bg=C_BG)
        self.root_container.pack(fill="both", expand=True)
        self.root_container.grid_rowconfigure(0, weight=1)
        self.root_container.grid_columnconfigure(0, weight=1)

        self.screen_unlock = tk.Frame(self.root_container, bg=C_BG)
        self.screen_main = tk.Frame(self.root_container, bg=C_BG)

        self.screen_unlock.grid(row=0, column=0, sticky="nsew")
        self.screen_main.grid(row=0, column=0, sticky="nsew")

        self._build_unlock_ui()
        self._build_main_ui()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.screen_unlock.tkraise()
        self._start_unlock_process()

    def _put_text_vn(self, cv2_im, text, position, font, color=(255, 255, 255)):
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    def _load_models(self):
        if os.path.exists(LBPH_MODEL):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(LBPH_MODEL)

    def _build_unlock_ui(self):
        tk.Label(
            self.screen_unlock,
            text="YÊU CẦU PHÁT HIỆN MẶT NGƯỜI ĐỂ MỞ KHÓA",
            font=("Segoe UI", 24, "bold"),
            bg=C_BG,
            fg=C_ACCENT,
        ).pack(pady=(50, 20))
        self.unlock_cam_lbl = tk.Label(
            self.screen_unlock, bg="#000", width=800, height=600
        )
        self.unlock_cam_lbl.pack(pady=20)
        self.unlock_msg = tk.StringVar(value="Hệ thống đang quét không gian...")
        tk.Label(
            self.screen_unlock,
            textvariable=self.unlock_msg,
            font=("Segoe UI", 16),
            bg=C_BG,
            fg=C_WARN,
        ).pack()

    def _start_unlock_process(self):
        self.app_state = "init"
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self._unlock_loop, daemon=True).start()

    def _unlock_loop(self):
        consecutive_frames = 0
        while self.app_state == "init":
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))

            if len(rects) > 0:
                consecutive_frames += 1
                for x, y, w, h in rects:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame = self._put_text_vn(
                        frame,
                        "Đã khóa mục tiêu",
                        (x, y - 25),
                        self.vn_font_small,
                        (0, 255, 0),
                    )
            else:
                consecutive_frames = 0

            if consecutive_frames > 0:
                self.unlock_msg.set(
                    f"Đã phát hiện đối tượng, đang xác nhận... ({consecutive_frames}/10)"
                )
            else:
                self.unlock_msg.set("Vui lòng đưa mặt vào khung hình camera...")

            display = cv2.resize(frame, (800, 600))
            self._display_frame(display, self.unlock_cam_lbl)

            if consecutive_frames >= 10:
                self.app_state = "main"
                break
            time.sleep(0.03)

        if self.cap:
            self.cap.release()
        self.after(500, self._switch_to_main_app)

    def _switch_to_main_app(self):
        self.screen_main.tkraise()
        self._set_mode("auth")
        self._show_placeholder()

    def _build_main_ui(self):
        hdr = tk.Frame(self.screen_main, bg=C_PANEL, pady=15)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="HỆ THỐNG NHẬN DẠNG VÀ ĐĂNG KÝ",
            font=("Segoe UI", 24, "bold"),
            bg=C_PANEL,
            fg=C_ACCENT,
        ).pack()
        menu_frm = tk.Frame(self.screen_main, bg=C_BG)
        menu_frm.pack(fill="x", pady=15, padx=30)

        self.btn_mode_auth = tk.Button(
            menu_frm,
            text="GIAO DIỆN XÁC THỰC",
            font=("Segoe UI", 12, "bold"),
            bg=C_GREEN,
            fg="#000",
            relief="flat",
            command=lambda: self._set_mode("auth"),
        )
        self.btn_mode_auth.pack(side="left", fill="x", expand=True, padx=10, ipady=10)

        self.btn_mode_reg = tk.Button(
            menu_frm,
            text="GIAO DIỆN ĐĂNG KÝ",
            font=("Segoe UI", 12, "bold"),
            bg=C_MUTED,
            fg="#fff",
            relief="flat",
            command=lambda: self._set_mode("register"),
        )
        self.btn_mode_reg.pack(side="right", fill="x", expand=True, padx=10, ipady=10)

        self.pages_container = tk.Frame(self.screen_main, bg=C_BG)
        self.pages_container.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        self.pages_container.grid_rowconfigure(0, weight=1)
        self.pages_container.grid_columnconfigure(0, weight=1)
        self.page_auth = tk.Frame(self.pages_container, bg=C_BG)
        self.page_reg = tk.Frame(self.pages_container, bg=C_BG)
        self.page_auth.grid(row=0, column=0, sticky="nsew")
        self.page_reg.grid(row=0, column=0, sticky="nsew")

        self._build_page_auth()
        self._build_page_reg()

    def _build_page_auth(self):
        self.page_auth.grid_columnconfigure(0, weight=1)
        self.page_auth.grid_columnconfigure(1, weight=0, minsize=400)
        self.page_auth.grid_rowconfigure(0, weight=1)
        cam_frame = tk.Frame(self.page_auth, bg="#000", bd=2, relief="sunken")
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        self.auth_cam_lbl = tk.Label(cam_frame, bg="#000")
        self.auth_cam_lbl.pack(expand=True)
        ctrl = tk.Frame(self.page_auth, bg=C_BG)
        ctrl.grid(row=0, column=1, sticky="nsew")

        self.btn_cam_auth = self._btn(
            ctrl, "▶ BẬT CAMERA XÁC THỰC", self._toggle_camera, C_ACCENT
        )
        self.btn_cam_auth.pack(fill="x", pady=(0, 15))

        frm_train = self._section(ctrl, "QUẢN LÝ MÔ HÌNH (LBPH)")
        self.btn_train = self._btn(
            frm_train, "⚙ CẬP NHẬT MODEL TỪ DATASET", self._train, "#7c4dff"
        )
        self.btn_train.pack(fill="x", pady=5)

        tk.Label(
            frm_train,
            text="Danh sách người dùng đã đăng ký:",
            font=("Segoe UI", 10),
            bg=C_PANEL,
            fg=C_MUTED,
        ).pack(anchor="w", pady=(5, 5))
        self.user_list = tk.Listbox(
            frm_train,
            font=("Consolas", 11),
            bg="#2a2d3a",
            fg=C_TEXT,
            relief="flat",
            height=10,
            bd=0,
        )
        self.user_list.pack(fill="both")
        self._refresh_user_list()

    def _build_page_reg(self):
        self.page_reg.grid_columnconfigure(0, weight=1)
        self.page_reg.grid_columnconfigure(1, weight=0, minsize=400)
        self.page_reg.grid_rowconfigure(0, weight=1)

        cam_frame = tk.Frame(self.page_reg, bg="#000", bd=2, relief="sunken")
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        self.reg_cam_lbl = tk.Label(cam_frame, bg="#000")
        self.reg_cam_lbl.pack(expand=True)

        self.reg_preview_lbl = tk.Label(cam_frame, bg="#000", bd=2, relief="solid")
        self.reg_preview_lbl.place(x=10, y=10)

        ctrl = tk.Frame(self.page_reg, bg=C_BG)
        ctrl.grid(row=0, column=1, sticky="nsew")

        self.btn_cam_reg = self._btn(
            ctrl, "▶ BẬT CAMERA ĐĂNG KÝ", self._toggle_camera, C_ACCENT
        )
        self.btn_cam_reg.pack(fill="x", pady=(0, 15))

        frm_register = self._section(ctrl, "THÔNG TIN NGƯỜI DÙNG MỚI")
        tk.Label(
            frm_register,
            text="Nhập họ và tên:",
            font=("Segoe UI", 12),
            bg=C_PANEL,
            fg=C_TEXT,
        ).pack(anchor="w")
        self.ent_name = tk.Entry(
            frm_register,
            font=("Segoe UI", 13),
            bg="#2a2d3a",
            fg=C_TEXT,
            relief="flat",
            bd=10,
        )
        self.ent_name.pack(fill="x", pady=(5, 20))

        self.progress_var = tk.IntVar(value=0)
        self.progress = ttk.Progressbar(
            frm_register, variable=self.progress_var, maximum=CAPTURE_COUNT
        )
        self.progress.pack(fill="x")
        self.prog_lbl = tk.StringVar(value=f"0 / {CAPTURE_COUNT} ảnh hoàn thành")
        tk.Label(
            frm_register,
            textvariable=self.prog_lbl,
            font=("Consolas", 10),
            bg=C_PANEL,
            fg=C_WARN,
        ).pack(anchor="w", pady=5)

        self.btn_reg = self._btn(
            frm_register, "📸 BẮT ĐẦU CHỤP HÌNH", self._start_capture, C_WARN
        )
        self.btn_reg.pack(fill="x", pady=10)

    def _set_mode(self, mode):
        self.mode = mode
        self.is_capturing = False

        self._show_placeholder()

        if mode == "auth":
            self.btn_mode_auth.configure(bg=C_GREEN, fg="#000")
            self.btn_mode_reg.configure(bg=C_MUTED, fg="#fff")
            self.page_auth.tkraise()
        else:
            self.btn_mode_auth.configure(bg=C_MUTED, fg="#fff")
            self.btn_mode_reg.configure(bg=C_WARN, fg="#000")
            self.page_reg.tkraise()
            self.progress_var.set(0)
            self.prog_lbl.set(f"0 / {CAPTURE_COUNT} ảnh hoàn thành")
            self.reg_preview_lbl.configure(image="")

    def _toggle_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.btn_cam_auth.configure(text="⏹ TẮT CAMERA", bg=C_RED)
            self.btn_cam_reg.configure(text="⏹ TẮT CAMERA", bg=C_RED)
            threading.Thread(target=self._main_camera_loop, daemon=True).start()
        else:
            self.running = False
            self.is_capturing = False
            time.sleep(0.2)
            if self.cap:
                self.cap.release()
            self.btn_cam_auth.configure(text="▶ BẬT CAMERA XÁC THỰC", bg=C_ACCENT)
            self.btn_cam_reg.configure(text="▶ BẬT CAMERA ĐĂNG KÝ", bg=C_ACCENT)
            self.reg_preview_lbl.configure(image="")
            self._show_placeholder()

    def _main_camera_loop(self):
        while self.running and self.app_state == "main":
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            self.frame_idx += 1
            self.quality_warning = ""
            processed_face_for_ui = None

            if self.frame_idx % (SKIP_FRAMES + 1) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_h, img_w = gray.shape

                rects = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                if len(rects) > 0:
                    rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
                    x, y, w, h = rects[0]
                    x1, y1, x2, y2 = _clamp_bbox_xyxy(
                        (x, y, x + w, y + h), img_w, img_h
                    )
                    self._primary_face = (x1, y1, x2, y2)

                    raw_crop = gray[y1:y2, x1:x2]
                    if raw_crop.size > 0:
                        is_good, warning = self._check_image_quality(raw_crop)
                        processed_face = apply_advanced_preprocessing(
                            raw_crop, self.face_mask, TARGET_SIZE
                        )
                        processed_face_for_ui = processed_face

                        if self.mode == "auth":
                            self._last_result = self._recognize(processed_face)
                        elif self.mode == "register" and self.is_capturing:
                            if is_good:
                                self._capture_logic(processed_face)
                            else:
                                self.quality_warning = warning
                else:
                    self._primary_face = None
                    self._last_result = None

            display = self._draw_overlay(display)
            display = cv2.resize(display, (800, 600))

            if self.mode == "auth":
                self._display_frame(display, self.auth_cam_lbl)
            else:
                self._display_frame(display, self.reg_cam_lbl)
                if processed_face_for_ui is not None:
                    prev_img = cv2.resize(processed_face_for_ui, (120, 120))
                    self._display_frame(prev_img, self.reg_preview_lbl, is_gray=True)
                else:
                    self.reg_preview_lbl.configure(image="")

    def _check_image_quality(self, gray_roi):
        blur_val = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        brightness = np.mean(gray_roi)
        if blur_val < BLUR_THRESHOLD:
            return False, "HÌNH QUÁ MỜ!"
        if brightness < BRIGHTNESS_MIN:
            return False, "QUÁ TỐI!"
        if brightness > BRIGHTNESS_MAX:
            return False, "CHÓI SÁNG!"
        return True, "OK"

    def _recognize(self, preprocessed_face):
        if self.recognizer is None or not self.label_map:
            return ("Chưa huấn luyện", 999, False)

        label, conf = self.recognizer.predict(preprocessed_face)

        ok = conf < CONFIDENCE_TH

        name = self.label_map.get(label, "Khách") if ok else "Không xác định"
        return (name, conf, ok)

    def _draw_overlay(self, display):
        if not self._primary_face:
            return display
        x1, y1, x2, y2 = self._primary_face

        if self.mode == "auth":
            if self._last_result:
                name, conf, ok = self._last_result
                color = (0, 255, 0) if ok else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                display = self._put_text_vn(
                    display,
                    f"{name} (Độ sai lệch: {conf:.0f})",
                    (x1, y1 - 25),
                    self.vn_font_small,
                    color,
                )

        elif self.mode == "register":
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if self.quality_warning:
                display = self._put_text_vn(
                    display,
                    self.quality_warning,
                    (x1, y1 - 25),
                    self.vn_font_small,
                    (0, 0, 255),
                )
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)

            if self.is_capturing:
                cv2.rectangle(display, (20, 20), (280, 60), (0, 0, 0), -1)
                display = self._put_text_vn(
                    display,
                    f"Tiến trình: {self.reg_count}/{CAPTURE_COUNT} ảnh",
                    (30, 25),
                    self.vn_font_small,
                    (255, 255, 255),
                )
        return display

    def _start_capture(self):
        name = self.ent_name.get().strip()
        if not name:
            return messagebox.showwarning("Lỗi", "Vui lòng nhập họ tên!")
        if not self.running:
            return messagebox.showwarning("Lỗi", "Vui lòng bật camera trước!")

        self.reg_id = next_label_id(self.label_map)
        self.reg_name = name
        self.reg_count = 0
        self.progress_var.set(0)

        self.label_map[self.reg_id] = name
        self.reg_folder = os.path.join(DATASET_DIR, str(self.reg_id))
        os.makedirs(self.reg_folder, exist_ok=True)

        self.is_capturing = True
        self.btn_reg.configure(state="disabled", text="⏸ ĐANG QUÉT DATA...")

    def _capture_logic(self, preprocessed_face):
        if self.reg_count >= CAPTURE_COUNT:
            return

        fname = os.path.join(self.reg_folder, f"{self.reg_count:03d}.jpg")
        cv2.imwrite(fname, preprocessed_face)
        self.reg_count += 1

        self.after(0, self._update_progress, self.reg_count)

        if self.reg_count >= CAPTURE_COUNT:
            self.is_capturing = False
            save_label_map(self.label_map)
            self.after(0, self._register_done)

    def _update_progress(self, count):
        self.progress_var.set(count)
        self.prog_lbl.set(f"{count} / {CAPTURE_COUNT} ảnh hoàn thành")

    def _register_done(self):
        self.btn_reg.configure(state="normal", text="📸 BẮT ĐẦU CHỤP")
        self._refresh_user_list()
        messagebox.showinfo(
            "Hoàn tất",
            f"Thu thập đủ {CAPTURE_COUNT} ảnh đạt chuẩn.\nVui lòng chuyển sang tab XÁC THỰC và bấm Cập nhật Model.",
        )
        self._set_mode("auth")

    def _train(self):
        self.btn_train.configure(text="⏳ ĐANG HUẤN LUYỆN...")
        self.update()
        ok, msg = train_lbph()
        if ok:
            self._load_models()
            messagebox.showinfo("Thành công", msg)
        self.btn_train.configure(text="⚙ CẬP NHẬT MODEL TỪ DATASET")

    def _section(self, parent, title):
        frm = tk.Frame(parent, bg=C_PANEL, padx=15, pady=15)
        frm.pack(fill="x", pady=(0, 15))
        tk.Label(
            frm, text=title, font=("Segoe UI", 12, "bold"), bg=C_PANEL, fg=C_ACCENT
        ).pack(anchor="w", pady=(0, 15))
        return frm

    def _btn(self, parent, text, cmd, color):
        return tk.Button(
            parent,
            text=text,
            command=cmd,
            font=("Segoe UI", 11, "bold"),
            bg=color,
            fg="#000",
            relief="flat",
            pady=10,
            cursor="hand2",
        )

    def _refresh_user_list(self):
        self.user_list.delete(0, "end")
        for lid, name in self.label_map.items():
            self.user_list.insert("end", f" [ID: {lid:02d}] {name}")

    def _show_placeholder(self):
        if not self.running:
            img = np.zeros((600, 800, 3), dtype=np.uint8)
            img = self._put_text_vn(
                img,
                "CAMERA ĐANG TẮT",
                (250, 280),
                self.vn_font_large,
                (100, 100, 120),
            )
            if self.mode == "auth":
                self._display_frame(img, self.auth_cam_lbl)
            else:
                self._display_frame(img, self.reg_cam_lbl)

    def _display_frame(self, img_array, label_widget, is_gray=False):
        if is_gray:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label_widget.imgtk = imgtk
        label_widget.configure(image=imgtk)

    def _on_close(self):
        self.running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = FaceAuthApp()
    app.mainloop()