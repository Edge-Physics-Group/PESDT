import sys, json, os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QCheckBox,
    QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QTabWidget, QSpinBox, QGridLayout, 
    QDoubleSpinBox, QScrollArea, QGroupBox, QFrame, QSizePolicy
)
import matplotlib
matplotlib.use('Qt5Agg')  # Or 'QtAgg' depending on your version
#

class CollapsibleBox(QWidget):
    def __init__(self, title=""):
        super().__init__()
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet("text-align: left;")
        self.toggle_button.clicked.connect(self.toggle_content)

        self.content_area = QWidget()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_layout = QVBoxLayout()
        self.content_area.setLayout(self.content_layout)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)
        self.setLayout(main_layout)

    def toggle_content(self):
        self.content_area.setVisible(self.toggle_button.isChecked())

    def layout(self):
        return self.content_layout

class Base(QWidget):
    def __init__(self, machine_dict = None):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.machine_diags = machine_dict
        # Machine dropdown
        machine_layout = QHBoxLayout()
        machine_label = QLabel("Machine:")
        self.machine_combo = QComboBox()
        self.machine_combo.addItems(self.machine_diags.keys())  # You can extend this later
        self.machine_combo.currentIndexChanged.connect(self.update_diagnostics)
        machine_layout.addWidget(machine_label)
        machine_layout.addWidget(self.machine_combo)
        layout.addLayout(machine_layout)

        # Pulse number (SpinBox for integers)
        pulse_layout = QHBoxLayout()
        pulse_label = QLabel("Pulse:")
        self.pulse_spin = QSpinBox()
        self.pulse_spin.setMaximum(999999)
        self.pulse_spin.setValue(81472)
        pulse_layout.addWidget(pulse_label)
        pulse_layout.addWidget(self.pulse_spin)
        layout.addLayout(pulse_layout)

        
        
        # Edge Code settings
        edge_code_code_layout = QHBoxLayout()
        edge_code_code_label = QLabel("Edge Code:")
        self.edge_code_combo = QComboBox()
        self.edge_code_combo.addItems(["edge2d", "solps", "oedge"])
        edge_code_code_layout.addWidget(edge_code_code_label)
        edge_code_code_layout.addWidget(self.edge_code_combo)
        layout.addLayout(edge_code_code_layout)

        edge_path_layout = QHBoxLayout()
        edge_path_label = QLabel("  Sim Path:")
        self.edge_path_input = QLineEdit("/home/mgroth/cmg/catalog/edge2d/jet/81472/jul3122/seq#1/tran")
        edge_path_layout.addWidget(edge_path_label)
        edge_path_layout.addWidget(self.edge_path_input)
        layout.addLayout(edge_path_layout)

        data_source_layout = QHBoxLayout()
        data_source_label = QLabel("Atomic and molecular data source:")
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["AMJUEL", "YACORA", "ADAS"])
        data_source_layout.addWidget(data_source_label)
        data_source_layout.addWidget(self.data_source_combo)
        layout.addLayout(data_source_layout)

        # Checkboxes
        self.read_adas_checkbox = QCheckBox("Read ADAS (Check if you are using ADAS data for the first time, or you've added new lines)")
        self.read_adas_checkbox.setChecked(False)
        layout.addWidget(self.read_adas_checkbox)

        self.recalc_h2_pos = QCheckBox(r"Re-calculate H2+ density (SOLPS, always on for EDEG2D and OEDGE)")
        self.recalc_h2_pos.setChecked(False)
        layout.addWidget(self.recalc_h2_pos)

        self.run_cherab = QCheckBox("Run Cherab")
        self.run_cherab.setChecked(False)
        layout.addWidget(self.run_cherab)

        self.analyse_synth_spec_features = QCheckBox(r"Calculate Continuum Te and Stark broadened ne")
        self.analyse_synth_spec_features.setChecked(False)
        layout.addWidget(self.analyse_synth_spec_features)

        

        # Save dir
        save_layout = QHBoxLayout()
        save_label = QLabel("Save Dir:")
        self.save_input = QLineEdit("PESDT_cases/")
        save_layout.addWidget(save_label)
        save_layout.addWidget(self.save_input)
        layout.addLayout(save_layout)

        # Diagnostics area (will hold grid of diagnostics)
        diag_layout = QVBoxLayout()
        diag_label = QLabel("Diagnostics:")
        diag_layout.addWidget(diag_label)

        self.diag_grid_widget = QWidget()
        self.diag_grid = QGridLayout()
        self.diag_grid.setSpacing(10)
        self.diag_grid_widget.setLayout(self.diag_grid)

        # Optional: make the diagnostic area scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.diag_grid_widget)

        diag_layout.addWidget(scroll_area)
        layout.addLayout(diag_layout)

        self.update_diagnostics()

    def update_diagnostics(self):
        # Clear the old diagnostics
        while self.diag_grid.count():
            item = self.diag_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        machine = self.machine_combo.currentText()
        diag_dict = self.machine_diags.get(machine, {})

        # Populate diagnostics in grid
        items = list(diag_dict.keys())
        columns = 3
        for index, diag in enumerate(items):
            row = index // columns
            col = index % columns

            diag_widget = QWidget()
            h_layout = QHBoxLayout()
            h_layout.setContentsMargins(5, 5, 5, 5)
            
            checkbox = QCheckBox()
            h_layout.addWidget(checkbox)
            h_layout.addWidget(QLabel(diag))
            h_layout.addStretch()
            diag_widget.setLayout(h_layout)

            self.diag_grid.addWidget(diag_widget, row, col)

    def get_selected_diagnostics(self):
        selected = []
        for i in range(self.diag_grid.count()):
            item = self.diag_grid.itemAt(i)
            widget = item.widget()
            if widget:
                layout = widget.layout()

                checkbox = layout.itemAt(0).widget()
                label_widget = layout.itemAt(1).widget()

                if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                    if isinstance(label_widget, QLabel):
                        selected.append(label_widget.text())
    
    def get_settings(self):
        return {
            "machine": self.machine_combo.currentText(),
            "pulse": self.pulse_spin.value(),
            "edge_code": {
                "code": self.edge_code_combo.currentText(),
                "sim_path": self.edge_path_input.text()
            },
            "read_ADAS": self.read_adas_checkbox.isChecked(),
            "save_dir": self.save_input.text(),
            "diag_list": self.get_selected_diagnostics(),
            "run_options": {
                "run_cherab": self.run_cherab.isChecked(),
                "analyse_synth_spec_features": self.analyse_synth_spec_features.isChecked(),
                "data_source": self.data_source_combo.currentText(),
                "recalc_h2_pos": self.recalc_h2_pos.isChecked(),
                "Sion_H_transition": [[2,1],[3,2]],
                "Srec_H_transition": [[5,2]]
            }
    }

    

class EmissionLines(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.setLayout(QVBoxLayout())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.content_layout = QVBoxLayout()
        content.setLayout(self.content_layout)
        scroll.setWidget(content)
        self.layout().addWidget(scroll)

        self.checkboxes = {}  # {(atom_num, wavelength): QCheckBox}

        self.populate()

    def populate(self):
        for element_name in ['H', 'He', 'C', 'Be', 'N', 'W']:
            data = getattr(self.db, f"{element_name}_lines", None)
            if not data:
                continue
            atom_num = data.get("ATOM_NUM")

            group_box = CollapsibleBox(f"{element_name} (Z={atom_num})")

            for series_name, lines in data.items():
                if series_name == "ATOM_NUM":
                    continue

                group_box.layout().addWidget(QLabel(f"{series_name}:"))

                line_row = QHBoxLayout()
                for wl, pn in lines.items():
                    box = QCheckBox(wl)
                    self.checkboxes[(atom_num, wl)] = (box, pn)
                    line_row.addWidget(box)

                container = QWidget()
                container.setLayout(line_row)
                group_box.layout().addWidget(container)

                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                group_box.layout().addWidget(separator)

            self.content_layout.addWidget(group_box)

        self.content_layout.addStretch()

    def get_selected_lines(self):
        result = {}
        for (atom_num, wl), (box, pn) in self.checkboxes.items():
            if box.isChecked():
                result.setdefault(atom_num, {})[wl] = pn
        return result

class CherabSettings(QWidget):
    def __init__(self, emission_lines_widget: EmissionLines):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        num_processes_layout = QHBoxLayout()
        num_processes_label = QLabel("Number of processes (threads): ")
        self.num_processes = QSpinBox()
        self.num_processes.setMaximum(24)
        self.num_processes.setValue(8)
        num_processes_layout.addWidget(num_processes_label)
        num_processes_layout.addWidget(self.num_processes)
        layout.addLayout(num_processes_layout)

        pixel_samples_layout = QHBoxLayout()
        pixel_samples_label = QLabel("Number of pixel samples (Number of MC rays):")
        self.pixel_samples = QSpinBox()
        self.pixel_samples.setMaximum(1000000)
        self.pixel_samples.setValue(1000)
        pixel_samples_layout.addWidget(pixel_samples_label)
        pixel_samples_layout.addWidget(self.pixel_samples)
        layout.addLayout(pixel_samples_layout)

        ray_extinction_prob_layout = QHBoxLayout()
        ray_extinction_prob_label = QLabel("Ray extinction probability (0,1]: ")
        self.ray_extinction_prob = QDoubleSpinBox()
        self.ray_extinction_prob.setMaximum(1.0)
        self.ray_extinction_prob.setMinimum(0.001)
        self.ray_extinction_prob.setValue(0.01)
        ray_extinction_prob_layout.addWidget(ray_extinction_prob_label)
        ray_extinction_prob_layout.addWidget(self.ray_extinction_prob)
        layout.addLayout(ray_extinction_prob_layout)        

        self.import_jet_surfaces = QCheckBox("Import full mesh (JET only)")
        self.import_jet_surfaces.setChecked(False)
        layout.addWidget(self.import_jet_surfaces)

        self.include_reflections = QCheckBox("Include reflections")
        self.include_reflections.setChecked(False)
        layout.addWidget(self.include_reflections)

        self.calculate_stark_ne = QCheckBox("Calculate Stark broadening (needed for ne estimate)")
        self.calculate_stark_ne.setChecked(False)
        layout.addWidget(self.calculate_stark_ne)

        self.ff_fb_emission = QCheckBox("Calculate continuum emission (300-500 nm, needed for Te estimate)")
        self.ff_fb_emission.setChecked(False)
        layout.addWidget(self.ff_fb_emission)

        self.mol_exc_emission = QCheckBox("Calculate molecular excitation emission")
        self.mol_exc_emission.setChecked(False)
        layout.addWidget(self.mol_exc_emission)

        mol_lines = ["lyman", "werner", "fulcher"]
        molecular_bands = QLabel("Molecular bands: ")
        layout.addWidget(molecular_bands)

        
        bands_layout = QHBoxLayout()
        layout.addLayout(bands_layout)

        self.molecular_bands_boxes = []

        for item in mol_lines:
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(0)  

            checkbox = QCheckBox(item)
            item_layout.addWidget(checkbox)

            bands_layout.addWidget(item_widget)
            self.molecular_bands_boxes.append((checkbox, item))

        stark_spectral_bins_layout = QHBoxLayout()
        stark_spectral_bins_label = QLabel("Number of spectral bins for Stark broadening:")
        self.stark_spectral_bins = QSpinBox()
        self.stark_spectral_bins.setMaximum(1000)
        self.stark_spectral_bins.setValue(50)
        stark_spectral_bins_layout.addWidget(stark_spectral_bins_label)
        stark_spectral_bins_layout.addWidget(self.stark_spectral_bins)
        layout.addLayout(stark_spectral_bins_layout)

        ff_fb_spectral_bins_layout = QHBoxLayout()
        ff_fb_spectral_bins_label = QLabel("Number of spectral bins for continuum emission:")
        self.ff_fb_spectral_bins = QSpinBox()
        self.ff_fb_spectral_bins.setMaximum(1000)
        self.ff_fb_spectral_bins.setValue(50)
        ff_fb_spectral_bins_layout.addWidget(ff_fb_spectral_bins_label)
        ff_fb_spectral_bins_layout.addWidget(self.ff_fb_spectral_bins)
        layout.addLayout(ff_fb_spectral_bins_layout)

        self.emission_lines = emission_lines_widget



        stark_transition = QLabel("Line for Stark transition:")
        layout.addWidget(stark_transition)

        self.stark_transition_combo = QComboBox()
        layout.addWidget(self.stark_transition_combo)

        self.refresh_btn = QPushButton("Refresh from selected lines")
        self.refresh_btn.clicked.connect(self.update_lines)
        layout.addWidget(self.refresh_btn)

    def update_lines(self):
        selected = self.emission_lines.get_selected_lines()
        self.stark_transition_combo.clear()
        # Flatten to a list of strings like: "1: 1215.2"
        for atom_num, lines in selected.items():
            for wl in lines.keys():
                transition = lines[wl]  # [p, n]
                label = f"{atom_num}: {wl}"
                self.stark_transition_combo.addItem(label, userData=transition)

    def get_selected(self):
        return [item for checkbox, item in self.molecular_bands_boxes if checkbox.isChecked()]

    def get_settings(self):
        return {
            "num_processes": self.num_processes.value(),
            "pixel_samples": self.pixel_samples.value(),
            "import_jet_surfaces": self.import_jet_surfaces.isChecked(),
            "include_reflections": self.include_reflections.isChecked(),
            "ray_extinction_prob": self.ray_extinction_prob.value(),
            "calculate_stark_ne": self.calculate_stark_ne.isChecked(),
            "stark_transition": self.stark_transition_combo.currentData(),
            "stark_spectral_bins": self.stark_spectral_bins.value(),
            "ff_fb_emission": self.ff_fb_emission.isChecked(),
            "ff_fb_spectral_bins": self.ff_fb_spectral_bins.value(),
            "mol_exc_emission": self.mol_exc_emission.isChecked(),
            "mol_exc_emission_bands": self.get_selected()
            
        }
class Main(QWidget):
    def __init__(self, machine_dict = None, spect_db = None):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("NOT YET FUNCTIONAL, USE PESDT_run.py")

        

        self.button = QPushButton("Submit job")
        self.button2 = QPushButton("Save input")
        self.button.clicked.connect(self.on_click)
        self.button2.clicked.connect(self.on_click2)
        self.tabs = QTabWidget()
        self.base_tab = Base(machine_dict = machine_dict)
        self.em_tab = EmissionLines(spect_db)
        self.cherab_tab = CherabSettings(self.em_tab)
        self.tabs.addTab(self.base_tab, "Run Settings")
        self.tabs.addTab(self.em_tab, "Emission lines")
        self.tabs.addTab(self.cherab_tab, "Cherab settings")
        layout.addWidget(self.tabs)
        layout.addWidget(self.label)
        input_layout = QHBoxLayout()
        input_label = QLabel("Input file path:")
        self.input_path = QLineEdit("PESDT_input/")
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_path)
        layout.addLayout(input_layout)
        layout.addWidget(self.button)
        layout.addWidget(self.button2)
        self.setLayout(layout)

    def on_click(self):
        pass
        #self.label.setText("Tab 1 Button Clicked!")
    def on_click2(self):
        settings_dict = self.base_tab.get_settings()
        settings_dict["cherab_options"] = self.cherab_tab.get_settings()
        settings_dict["spec_line_dict"] = {"1": self.em_tab.get_selected_lines()}

        # Get full path from input field
        save_path = os.path.expanduser(self.input_path.text())
        save_path = os.path.abspath(save_path)

        # Ensure the parent directory exists
        save_dir = os.path.dirname(save_path)
        Path(save_dir).mkdir(parents=True, exist_ok=True)


        with open(save_path, "w") as f:
            json.dump(settings_dict, f, indent=2)

        #self.label.setText("Tab 1 Button Clicked!")
class PostProcess(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        #self.label = QLabel("Post-processor")
        self.button = QPushButton("Plot")
        self.button.clicked.connect(self.on_click)
        #layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def on_click(self):
        pass
        #self.label.setText("Tab 2 Button Clicked!")

class PESDTGui(QWidget):
    def __init__(self, machine_dict = None, spect_db = None):
        super().__init__()

        self.setWindowTitle("PESDT 2.0")
        self.setGeometry(100, 100, 300, 300)
        self.tabs = QTabWidget()
        self.tabs.addTab(Main(machine_dict = machine_dict, spect_db = spect_db), "Main")
        self.tabs.addTab(PostProcess(), "Post-processor")

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    from core.utils import get_JETdefs, get_DIIIDdefs
    from core.database import spectroscopic_lines_db
    jet_dict = get_JETdefs().diag_dict
    dIIId_dict = get_DIIIDdefs().diag_dict
    machine_dict = {
            "JET":   jet_dict,
            "DIII-D": dIIId_dict
        }
    window = PESDTGui(machine_dict = machine_dict, spect_db = spectroscopic_lines_db())
    
    window.show()
    sys.exit(app.exec())