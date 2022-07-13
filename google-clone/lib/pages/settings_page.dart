import 'package:flutter/material.dart';
import 'package:google_clone/provider/main_screen_provider.dart';
import 'package:google_clone/provider/settings_screen_provider.dart';
import 'package:provider/provider.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({Key? key}) : super(key: key);

  @override
  _SettingsPageState createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  @override
  Widget build(BuildContext context) {
    final mainScreenProvider = Provider.of<MainScreenProvider>(context);
    final settingsScreenProvider = Provider.of<SettingsScreenProvider>(context);
    return Container(
      child: Scaffold(
        backgroundColor: Colors.white,
        body: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
          mainScreenProvider.getHeader(context),
          SizedBox(height: 10),
          Container(
              child: FutureBuilder(
                  future: Future.wait([
                    settingsScreenProvider.appConfig.getHost().then((value) {
                      settingsScreenProvider.hostTextController.text = value;
                      return value;
                    }), // String
                    settingsScreenProvider.appConfig.getPort().then((value) {
                      settingsScreenProvider.portTextController.text =
                          value.toString();
                      return value.toString();
                    }),
                    settingsScreenProvider.appConfig.getScheme().then((value) {
                      settingsScreenProvider.schemeTextController.text = value;
                      return value;
                    }) // Int
                  ]),
                  builder: (context, AsyncSnapshot<List<String>> snapshot) =>
                      Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text("Settings"),
                          _TextInputField(
                              title: "Host",
                              textController: settingsScreenProvider
                                  .hostTextController),
                          _TextInputField(
                              title: "Port",
                              textController: settingsScreenProvider
                                  .portTextController),
                          _TextInputField(
                              title: "Scheme",
                              textController: settingsScreenProvider
                                  .schemeTextController),
                          ElevatedButton(
                              onPressed: () {
                                settingsScreenProvider.appConfig.setHost(
                                    settingsScreenProvider
                                        .hostTextController.text);
                                settingsScreenProvider.appConfig.setPort(
                                    int.parse(settingsScreenProvider
                                        .portTextController.text));
                                Navigator.of(context).pop();
                              },
                              child: Text("Save Changes")),
                        ],
                      )))
        ]),
      ),
    );
  }
}

class _TextInputField extends StatefulWidget {
  final TextEditingController textController;
  final String title;

  const _TextInputField(
      {Key? key, required this.title, required this.textController})
      : super(key: key);

  @override
  __TextInputFieldState createState() => __TextInputFieldState();
}

class __TextInputFieldState extends State<_TextInputField> {
  @override
  Widget build(BuildContext context) {
    return Container(
      child: Card(
        child: Row(
          children: [
            Text(widget.title),
            Flexible(child: TextField(controller: widget.textController))
          ],
        ),
      ),
    );
  }
}
