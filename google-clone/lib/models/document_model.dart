import 'package:json_annotation/json_annotation.dart';

part 'document_model.g.dart';

@JsonSerializable()
class DocumentModel {

  final String documentName;
  final String documentDir;

  DocumentModel({required this.documentName, required this.documentDir});

  factory DocumentModel.fromJson(Map<String, dynamic> json) => _$DocumentModelFromJson(json);

  Map<String, dynamic> toJson() => _$DocumentModelToJson(this);

}