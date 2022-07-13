// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'query_response_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

QueryResponseModel _$QueryResponseModelFromJson(Map<String, dynamic> json) =>
    QueryResponseModel(
      documents: (json['documents'] as List<dynamic>)
          .map((e) => DocumentModel.fromJson(e as Map<String, dynamic>))
          .toList(),
      responseTime: json['responseTime'] as int,
    );

Map<String, dynamic> _$QueryResponseModelToJson(QueryResponseModel instance) =>
    <String, dynamic>{
      'documents': instance.documents,
      'responseTime': instance.responseTime,
    };
