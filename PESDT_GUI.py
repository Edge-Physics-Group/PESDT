import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QCheckBox,
    QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QTabWidget, QSpinBox, QAbstractItemView, QListWidget
)

from core.utils import get_JETdefs, get_DIIIDdefs
test_dict = {"KT3A": 0, "KS3": 0, "KT1V": 0}
test_dict2 = {"VIS": 0, "VUV": 0, "TEST": 0}

class Base(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.machine_diags = {
            "JET": get_JETdefs().diag_dict() ,
            "DIII-D": get_DIIIDdefs().diag_dict()}

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

        # diag_list dropdown populated from dictionary keys
        diag_layout = QHBoxLayout()
        diag_label = QLabel("Diagnostic:")
        self.diag_list = QListWidget()
        self.diag_list.setSelectionMode(QAbstractItemView.MultiSelection)
        #self.diag_combo.addItems(test_dict)
        self.update_diagnostics()
        diag_layout.addWidget(diag_label)
        diag_layout.addWidget(self.diag_list)
        layout.addLayout(diag_layout)

    def update_diagnostics(self):
        machine = self.machine_combo.currentText()
        diag_dict = self.machine_diags.get(machine, {})
        
        self.diag_list.clear()
        self.diag_list.addItems(diag_dict.keys())

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

    def get_selected_diagnostics(self):
        return [item.text() for item in self.diag_list.selectedItems()]

class EmissionLines(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

class CherabSettings(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)        

class Main(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("NOT YET FUNCTIONAL, USE PESDT_run.py")
        self.button = QPushButton("Submit job")
        self.button.clicked.connect(self.on_click)
        self.tabs = QTabWidget()
        self.tabs.addTab(Base(), "Run Settings")
        self.tabs.addTab(EmissionLines(), "Emission lines")
        self.tabs.addTab(CherabSettings(), "Cherab settings")
        layout.addWidget(self.tabs)
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def on_click(self):
        pass
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
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PESDT 2.0")
        self.setGeometry(1000, 1000, 1000, 1000)
        self.tabs = QTabWidget()
        self.tabs.addTab(Main(), "Main")
        self.tabs.addTab(PostProcess(), "Post-processor")

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PESDTGui()
    window.show()
    sys.exit(app.exec())