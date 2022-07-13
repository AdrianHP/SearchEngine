import 'package:flutter/material.dart';


 List<String> querySug = <String>[
   'Africa',
   'Antarctica',
    'Asia',
     'Australia',
    'Europe',
     'North America',
    'South America',
  ];
  
  class AutoComplete extends StatefulWidget {
  
    @override
    State<StatefulWidget> createState() => _AutoCompleteState();
  }
  
  class _AutoCompleteState extends State<AutoComplete> {
  
    @override
    Widget build(BuildContext context) {
      return Padding(
        padding: EdgeInsets.all(15.0),
        child: Autocomplete<String>(
          optionsBuilder: (TextEditingValue textEditingValue) {
            return querySug.toList();
              // .where((String query) => continent.name.toLowerCase()
              //   .startsWith(textEditingValue.text.toLowerCase())
              // )
              // .toList();
          },
          fieldViewBuilder: (
            BuildContext context,
            TextEditingController fieldTextEditingController,
            FocusNode fieldFocusNode,
            VoidCallback onFieldSubmitted
          ) {
            return TextField(
              controller: fieldTextEditingController,
              focusNode: fieldFocusNode,
              style: const TextStyle(fontWeight: FontWeight.bold),
            );
          },
          onSelected: (String selection) {
            print('Selected: ${selection}');
          },
          optionsViewBuilder: (
            BuildContext context,
            AutocompleteOnSelected<String> onSelected,
            Iterable<String> options
          ) {
            return Align(
              alignment: Alignment.topLeft,
              child: Material(
                child: Container(
                  width: 300,
                  color: Colors.white,
                  child: ListView.builder(
                    padding: EdgeInsets.all(10.0),
                    itemCount: options.length,
                    itemBuilder: (BuildContext context, int index) {
                      final String option = options.elementAt(index);
  
                      return GestureDetector(
                        onTap: () {
                          onSelected(option);
                        },
                        child: ListTile(
                          title: Text(option, style: const TextStyle(color: Colors.black)),
                        ),
                      );
                    },
                  ),
                ),
              ),
            );
          },
        ),
      );
    }
  }