import sys
import csv
import struct
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QHBoxLayout, QDoubleSpinBox, QCheckBox, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def parse_bin_file(file_path):
    record_format = '<4B I H Q h h h h i i h I I H 4B'
    record_size = struct.calcsize(record_format)
    records = []
    with open(file_path, 'rb') as f:
        f.read(8)
        while True:
            chunk = f.read(record_size)
            if len(chunk) < record_size:
                break
            values = struct.unpack(record_format, chunk)
            records.append(values)
    return records

def save_to_csv(records, bin_filename):
    csv_filename = Path(bin_filename).with_suffix('.csv')
    header = [
        'title1_4', 'title2_4', 'title3_4', 'title4_4',
        'deviceId', 'channelId', 'timestamp',
        'cfd_y1', 'cfd_y2', 'heigth', 'baseline',
        'qLong', 'qShort', 'psdValue',
        'eventCounter', 'eventCounterPSD', 'decimationFactor',
        'postfix1', 'postfix2', 'postfix3', 'postfix4'
    ]
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in records:
            writer.writerow(row)
    return csv_filename

def read_qlong_qshort(csv_file):
    qlong, qshort = [], []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ql = int(row['qLong'])
                qs = int(row['qShort'])
                if ql > 0 and qs >= 0:
                    qlong.append(ql)
                    qshort.append(qs)
            except (ValueError, KeyError):
                continue
    return np.array(qlong), np.array(qshort)

class PSDApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSD GUI Viewer")
        self.resize(1200, 800)

        self.csv_file = None
        self.qlong = np.array([])
        self.qshort = np.array([])
        self.last_spectrum_data = ([], [])
        self.last_psd_data = []

        layout = QVBoxLayout()
        self.setLayout(layout)

        # === Загрузка файла ===
        self.load_button = QPushButton("Загрузить BIN файл")
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)

        # === Фильтрация PSD ===
        filter_layout = QHBoxLayout()
        self.filter_toggle = QCheckBox("Включить фильтр PSD")
        self.filter_toggle.setChecked(True)
        self.filter_toggle.stateChanged.connect(self.update_plots)
        filter_layout.addWidget(self.filter_toggle)

        filter_layout.addWidget(QLabel("Порог PSD:"))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setDecimals(4)
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setValue(0.1474)
        self.threshold_input.valueChanged.connect(self.update_plots)
        filter_layout.addWidget(self.threshold_input)

        layout.addLayout(filter_layout)

        # === Кнопки сохранения ===
        save_layout = QHBoxLayout()
        self.save_spectrum_btn = QPushButton("Сохранить спектр в CSV")
        self.save_spectrum_btn.clicked.connect(self.save_spectrum_data)
        save_layout.addWidget(self.save_spectrum_btn)

        self.save_psd_btn = QPushButton("Сохранить PSD в CSV")
        self.save_psd_btn.clicked.connect(self.save_psd_data)
        save_layout.addWidget(self.save_psd_btn)

        layout.addLayout(save_layout)

        # === График спектра ===
        self.figure_spectrum, self.ax_spectrum = plt.subplots()
        self.canvas_spectrum = FigureCanvas(self.figure_spectrum)
        layout.addWidget(QLabel("Спектр отклика"))
        layout.addWidget(self.canvas_spectrum)

        # === График PSD ===
        self.figure_psd, self.ax_psd = plt.subplots()
        self.canvas_psd = FigureCanvas(self.figure_psd)
        layout.addWidget(QLabel("PSD-диаграмма"))
        layout.addWidget(self.canvas_psd)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите BIN файл", "", "BIN Files (*.bin)")
        if not path:
            return
        records = parse_bin_file(path)
        self.csv_file = save_to_csv(records, path)
        self.qlong, self.qshort = read_qlong_qshort(self.csv_file)
        self.update_plots()

    def update_plots(self):
        if self.qlong.size == 0:
            return

        psd = 1 - (self.qshort / self.qlong)
        mask_valid = np.isfinite(psd)
        psd = psd[mask_valid]
        qlong_valid = self.qlong[mask_valid]

        if self.filter_toggle.isChecked():
            threshold = self.threshold_input.value()
            mask = psd > threshold
            psd_filtered = psd[mask]
            qlong_filtered = qlong_valid[mask]
        else:
            psd_filtered = psd
            qlong_filtered = qlong_valid

        # === Спектр ===
        self.ax_spectrum.clear()
        if qlong_filtered.size > 0:
            bin_edges = np.linspace(0, 100_000, 4097)
            counts, _ = np.histogram(qlong_filtered, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            self.ax_spectrum.plot(bin_centers, counts, color='blue')
            self.ax_spectrum.set_yscale('log')
            self.ax_spectrum.set_xlabel("qLong")
            self.ax_spectrum.set_ylabel("Количество событий")
            self.ax_spectrum.grid(True, linestyle='--', alpha=0.5)
            self.last_spectrum_data = (bin_centers, counts)
        else:
            self.ax_spectrum.text(0.5, 0.5, "Нет данных", ha='center', va='center')
            self.last_spectrum_data = ([], [])

        self.canvas_spectrum.draw()

        # === PSD ===
        self.ax_psd.clear()
        if psd_filtered.size > 0:
            self.ax_psd.hist(psd_filtered, bins=200, range=(0, 1), edgecolor='black')
            self.ax_psd.set_xlabel("PSD")
            self.ax_psd.set_ylabel("Количество событий")
            self.ax_psd.grid(True, linestyle='--', alpha=0.5)
            self.last_psd_data = psd_filtered
        else:
            self.ax_psd.text(0.5, 0.5, "Нет данных", ha='center', va='center')
            self.last_psd_data = []

        self.canvas_psd.draw()

    def save_spectrum_data(self):
        if not self.last_spectrum_data[0].any():
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить спектр", "spectrum_data.csv", "CSV Files (*.csv)")
        if not file_path:
            return
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['BinCenter', 'Count'])
            for x, y in zip(*self.last_spectrum_data):
                writer.writerow([x, y])

    def save_psd_data(self):
        if not len(self.last_psd_data):
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить PSD", "psd_data.csv", "CSV Files (*.csv)")
        if not file_path:
            return
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['PSD'])
            for value in self.last_psd_data:
                writer.writerow([value])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PSDApp()
    window.show()
    sys.exit(app.exec_())
