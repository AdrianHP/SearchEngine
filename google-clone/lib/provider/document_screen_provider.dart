

import 'package:flutter/material.dart';
import 'package:google_clone/services/api_service.dart';

class DocumentScreenProvider extends ChangeNotifier {

  final apiService = ApiService();

  String? _currentDocumentContent;
  String? get currentDocumentContent => _currentDocumentContent;

  Future<String?> getCurrentDocumentContent(BuildContext context, String currentDocumentDir) {
    return apiService.fetchDocument(context: context, documentDir: currentDocumentDir).then((value) {
      _currentDocumentContent = value;
      if (value != _currentDocumentContent) {
        notifyListeners();
      }
      return value;
    });
  }

}