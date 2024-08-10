import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart'; 
import 'dart:typed_data';
import 'dart:async';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Spectral Insight',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        textTheme: const TextTheme(
          bodyMedium: TextStyle(fontSize: 16, color: Colors.white),
        ),
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage>
    with SingleTickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final _wavelengthController = TextEditingController();
  final _frequencyController = TextEditingController();
  final _temperatureController = TextEditingController();
  final _radianceController = TextEditingController();
  bool _isLoading = false;
  String _result = '';

  // Animation controller for rotating the Earth
  late final AnimationController _controller;
  late final Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(seconds: 30),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0, end: 2 * 3.14159).animate(_controller);
    _controller.repeat();
    loadModel();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> loadModel() async {
    try {
      String? res = await Tflite.loadModel(
        model: "assets/model.tflite",
      );
      print(res);
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<void> predict() async {
    setState(() {
      _isLoading = true;
    });

    try {
      double wavelength = double.parse(_wavelengthController.text);
      double frequency = double.parse(_frequencyController.text);
      double temperature = double.parse(_temperatureController.text);
      double radiance = double.parse(_radianceController.text);

      List<double> input = [wavelength, frequency, temperature, radiance];
      var inputBuffer = Float32List.fromList(input).buffer.asUint8List();

      var output = await Tflite.runModelOnBuffer(
        buffer: inputBuffer,
        numThreads: 1,
        asynch: true,
      );

      setState(() {
        _result = output.toString();
      });
    } catch (e) {
      print("Error during prediction: $e");
      setState(() {
        _result = 'Error during prediction';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background with rotating Earth
          Positioned.fill(
            child: AnimatedBuilder(
              animation: _animation,
              builder: (context, child) {
                return Transform.rotate(
                  angle: _animation.value,
                  child: Image.asset(
                    'assets/earth.gif',
                    fit: BoxFit.cover,
                  ),
                );
              },
            ),
          ),
          // Foreground content
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _buildTextField(_wavelengthController, 'Wavelength (nm)'),
                const SizedBox(height: 10),
                _buildTextField(_frequencyController, 'Frequency (kHz)'),
                const SizedBox(height: 10),
                _buildTextField(_temperatureController, 'Temperature (°C)'),
                const SizedBox(height: 10),
                _buildTextField(_radianceController, 'Radiance'),
                const SizedBox(height: 20),
                _isLoading
                    ? Center(
                        child: CircularProgressIndicator(
                          valueColor: AlwaysStoppedAnimation<Color>(
                              Theme.of(context).primaryColor),
                        ),
                      )
                    : ElevatedButton(
                        onPressed: predict,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Theme.of(context).primaryColor,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                          padding: const EdgeInsets.symmetric(vertical: 15),
                        ),
                        child: const Text('Submit'),
                      ),
                const SizedBox(height: 20),
                Text(
                  'Prediction Result: $_result',
                  style: const TextStyle(fontSize: 16, color: Colors.white),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  TextFormField _buildTextField(
      TextEditingController controller, String label) {
    return TextFormField(
      controller: controller,
      keyboardType: TextInputType.number,
      decoration: InputDecoration(
        labelText: label,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide:
              BorderSide(color: Theme.of(context).primaryColor, width: 2),
        ),
      ),
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter a value';
        }
        return null;
      },
    );
  }
}
