import 'package:flutter/material.dart';
import 'package:google_clone/services/api_service.dart';

class SearchResultProvider extends ChangeNotifier {
  final apiService = ApiService();
  Map<String, dynamic>? searchResult;

  Future applyFeedback(
      BuildContext context, String query, String document, bool positive) {
    return apiService.applyFeedback(
        context: context,
        query: query,
        relevantDocumentsDirs: [if (positive) document],
        notRelevantDocumentsDirs: [if (!positive) document]);
  }
}
