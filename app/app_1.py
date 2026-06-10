import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import os, json, time, threading
from PIL import Image, ImageTk, ImageDraw, ImageFont


def _clamp_bbox_xyxy(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))
    if x2 <= x1:
        x2 = min(image_width, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height, y1 + 1)
    return [x1, y1, x2, y2]


def create_face_mask(shape=(250, 250)):
    r, c = shape
    mask = np.zeros((r, c), dtype=np.float64)
    center_x = int(c / 2)
    center_y = int(r / 1.8)
    axis_x = int(c / 2.6)
    axis_y = int(r / 2)
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    mask = np.power(mask, 1.5)
    return mask


def apply_advanced_preprocessing(gray_crop, mask, target_size=(250, 250)):
    face_resized = cv2.resize(gray_crop, target_size)
    face_denoised = cv2.bilateralFilter(face_resized, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_clahe = clahe.apply(face_denoised)
    face_processed = face_clahe.astype(np.float64) * mask
    return face_processed.astype(np.uint8)

