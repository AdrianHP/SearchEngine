import 'package:flutter/material.dart';

class SwitchButton extends StatefulWidget {
  final String title;
  final Future<bool> Function() onChange;
  late final bool _isActive;

  SwitchButton({required this.title, required bool isActive, required this.onChange}) {
    _isActive = isActive;
  }

  @override
  State<StatefulWidget> createState() {
    return _SwitchButtonState(isActive: _isActive);
  }
}

class _SwitchButtonState extends State<SwitchButton> {
  bool isActive = false;

  _SwitchButtonState({required bool isActive}) {
    this.isActive = isActive;
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Row(
        children: [
          Text(widget.title),
          Switch(
              value: isActive,
              onChanged: (bool value) {
                widget.onChange().then((value) => {
                      setState(() {
                        isActive = value;
                      })
                    });
              })
        ],
      ),
    );
  }
}
