import 'package:flutter/material.dart';
import 'package:google_clone/provider/document_screen_provider.dart';
import 'package:google_clone/provider/main_screen_provider.dart';
import 'package:provider/provider.dart';

class DocumentPage extends StatefulWidget {

  final String documentDir;

  const DocumentPage({ Key? key, required this.documentDir }) : super(key: key);

  @override
  State<DocumentPage> createState() => _DocumentPageState();
}

class _DocumentPageState extends State<DocumentPage> {
  @override
  Widget build(BuildContext context) {
    final mainScreenProvider = Provider.of<MainScreenProvider>(context);
    final documentScreenProvider = Provider.of<DocumentScreenProvider>(context);
    return Container(
      child: Scaffold(
        backgroundColor: Colors.white,
        body: Column(mainAxisSize: MainAxisSize.min, children: [
          mainScreenProvider.getHeader(context),
          SizedBox(height: 10),
          Container(
            child: FutureBuilder<String?>(future: 
              documentScreenProvider.getCurrentDocumentContent(context, widget.documentDir),
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  return Text(snapshot.data!);
                } else if (snapshot.hasError) {
                  return Text(snapshot.error.toString());
                }
                return Text("Loading...");
              },
            ),
          ),
        ]),
      )
    );
  }
}
