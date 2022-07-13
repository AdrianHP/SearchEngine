

import 'package:flutter/material.dart';
import 'package:google_clone/services/api_configuration_service.dart';

class SettingsScreenProvider extends ChangeNotifier {
  final appConfig = ApiConfigurationService();

  final hostTextController = TextEditingController();
  final portTextController = TextEditingController();
  final schemeTextController = TextEditingController();

  SettingsScreenProvider() {
    // hostTextController.addListener(() { notifyListeners(); });
    // portTextController.addListener(() { notifyListeners(); });
  }

}