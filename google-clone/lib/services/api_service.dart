import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:google_clone/models/query_response_model.dart';
import 'package:google_clone/services/api_configuration_service.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

class ApiService {
  bool useDummyData = false;
  static const _getQueryPath = "query";
  static const _getDocumentPath = "document";
  static const _applyFeedbackPath = "feedback";
  static const _getQueryExpand = "expand";

  
  Future<QueryResponseModel?> fetchData(
      {required BuildContext context,
      required String query,
      required int offset}) async {
    if (!this.useDummyData) {
      final apiConfigurationService =
          Provider.of<ApiConfigurationService>(context);
      final uri = await apiConfigurationService.getUrl(_getQueryPath,
          queryParams: {"query": query, "offset": offset.toString()});

      try
      {
         final response = await http.get(uri, headers: {
           'Content-type': 'application/json',
           "Access-Control-Allow-Origin": "*", // Required for CORS support to work
           "Access-Control-Allow-Credentials":
           'true', // Required for cookies, authorization headers with HTTPS
           "Access-Control-Allow-Headers":
           "Origin,Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,locale",
           "Access-Control-Allow-Methods": "POST, GET,OPTIONS"
        });
        Map<String, dynamic> res= jsonDecode(response.body);
        var result = QueryResponseModel.fromJson(res);
        return result;
      }
     catch (e) {
       print(e.toString());
      }
    }
    return null;
  }




  Future<String?> fetchDocument(
      {required BuildContext context,
      required String documentDir}) async {
    if (!this.useDummyData) {
      final apiConfigurationService =
          Provider.of<ApiConfigurationService>(context);
      final uri = await apiConfigurationService.getUrl(_getDocumentPath,
          queryParams: {"document_dir": documentDir});
      final response = await http.get(uri);
      return response.body;
    }
    return null;
  }

  Future<List<String>> fetchQuery(
      {required BuildContext context,
      required String query}) async {
    if (!this.useDummyData) {
      final apiConfigurationService =
          Provider.of<ApiConfigurationService>(context);
      final uri = await apiConfigurationService.getUrl(_getQueryExpand,
          queryParams: {"query": query});
      final response = await http.get(uri);
       try
      {
         final response = await http.get(uri, headers: {
           'Content-type': 'application/json',
           "Access-Control-Allow-Origin": "*", // Required for CORS support to work
           "Access-Control-Allow-Credentials":
           'true', // Required for cookies, authorization headers with HTTPS
           "Access-Control-Allow-Headers":
           "Origin,Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,locale",
           "Access-Control-Allow-Methods": "POST, GET,OPTIONS"
        });
          List<String> result = List<String>.from(json.decode(response.body));
          return result;
      }
     catch (e) {
       print(e.toString());
      }



       List<String> result = List<String>.from(json.decode(response.body));
        return result;
    }
    return [];
  }


  Future<bool> applyFeedback({
    required BuildContext context,
    required String query,
    required List<String> relevantDocumentsDirs,
    required List<String> notRelevantDocumentsDirs
  }) async {
      final apiConfigurationService =
          Provider.of<ApiConfigurationService>(context, listen: false);
      final uri = await apiConfigurationService.getUrl(_applyFeedbackPath);
      final json = {"query": query, "relevants": relevantDocumentsDirs, "not_relevants": notRelevantDocumentsDirs};
      final jsonBody = jsonEncode(json);
      final response = await http.post(uri, headers: {"Content-Type": "application/json"}, body: jsonBody);
      return response.statusCode == 200;
  }
}
