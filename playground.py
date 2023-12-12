from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QScrollArea, QSizePolicy, QSpacerItem, QProgressBar, QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QMenu, QMenuBar, QStatusBar, QToolBar, QSlider, QDial, QLCDNumber, QSplitter, QFrame

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up main window and layout
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add tabs
        self.tabs.addTab(self.trainingTab(), "Training")
        self.tabs.addTab(self.testingTab(), "Testing")

        # TODO Other UI setup like menu, status bar...

        # TODO get config from TOML file and populate UI fields

        # TODO set up model path and dataset path fields to open file dialog

    def trainingTab(self):
        # Create and return training tab widget
        # Create a widget and a layout
        tab = QWidget()
        layout = QGridLayout()

        # Add input fields
        self.batchSizeInput = QSpinBox()
        layout.addWidget(QLabel("Batch Size:"), 0, 0)
        layout.addWidget(self.batchSizeInput, 0, 1)
        layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochsInput = QSpinBox()
        layout.addWidget(self.epochsInput, 1, 1)
        layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.learningRateInput = QDoubleSpinBox()
        layout.addWidget(self.learningRateInput, 2, 1)
        layout.addWidget(QLabel("Momentum:"), 3, 0)
        self.momentumInput = QDoubleSpinBox()
        layout.addWidget(self.momentumInput, 3, 1)
        layout.addWidget(QLabel("Weight Decay:"), 4, 0)
        self.weightDecayInput = QDoubleSpinBox()
        layout.addWidget(self.weightDecayInput, 4, 1)
        layout.addWidget(QLabel("Optimizer:"), 5, 0)
        self.optimizerInput = QComboBox()
        self.optimizerInput.addItems(["SGD", "Adam"])
        layout.addWidget(self.optimizerInput, 5, 1)
        layout.addWidget(QLabel("Loss Function:"), 6, 0)
        self.lossFunctionInput = QComboBox()
        self.lossFunctionInput.addItems(["MSE", "Cross Entropy"])
        layout.addWidget(self.lossFunctionInput, 6, 1)
        layout.addWidget(QLabel("Model:"), 7, 0)
        self.modelInput = QComboBox()
        self.modelInput.addItems(["ViT", "ResNet", "AlexNet"])
        layout.addWidget(self.modelInput, 7, 1)
        layout.addWidget(QLabel("Dataset:"), 8, 0)
        self.datasetInput = QComboBox()
        self.datasetInput.addItems(["DRIVE", "STARE", "CHASE_DB1"])
        layout.addWidget(self.datasetInput, 8, 1)
        layout.addWidget(QLabel("Pretrained Model:"), 9, 0)
        self.pretrainedModelInput = QComboBox()
        self.pretrainedModelInput.addItems(["None", "ViT", "ResNet", "AlexNet"])
        layout.addWidget(self.pretrainedModelInput, 9, 1)
        layout.addWidget(QLabel("Pretrained Dataset:"), 10, 0)
        self.pretrainedDatasetInput = QComboBox()
        self.pretrainedDatasetInput.addItems(["DRIVE", "STARE", "CHASE_DB1"])
        layout.addWidget(self.pretrainedDatasetInput, 10, 1)
        layout.addWidget(QLabel("Pretrained Model Path:"), 11, 0)
        self.pretrainedModelPathInput = QLineEdit()
        layout.addWidget(self.pretrainedModelPathInput, 11, 1)
        layout.addWidget(QLabel("Pretrained Dataset Path:"), 12, 0)
        self.pretrainedDatasetPathInput = QLineEdit()
        layout.addWidget(self.pretrainedDatasetPathInput, 12, 1)
        layout.addWidget(QLabel("Output Path:"), 13, 0)
        self.outputPathInput = QLineEdit()
        layout.addWidget(self.outputPathInput, 13, 1)
        layout.addWidget(QLabel("GPU:"), 14, 0)
        self.gpuInput = QComboBox()
        self.gpuInput.addItems(["0", "1", "2"])
        layout.addWidget(self.gpuInput, 14, 1)



        # TODO Repeat for other fields...

        # Add Save Button
        saveButton = QPushButton("Save Configuration")
        saveButton.clicked.connect(self.saveTrainingConfig)
        layout.addWidget(saveButton)

        # Set the layout to the tab
        tab.setLayout(layout)
        return tab

        ...

    def testingTab(self):
        # TODO Create and return testing tab widget
        ...

    def saveTrainingConfig(self):
        # TODO  Function to save training config to TOML
        ...

    def startTraining(self):
        # TODO  Function to start training process
        ...

# main
if __name__ == "__main__":
    app = QApplication([])
    mainWindow = MainApp()
    mainWindow.show()
    app.exec()

