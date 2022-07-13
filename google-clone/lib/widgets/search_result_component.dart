import 'package:flutter/material.dart';

class SearchResultComponent extends StatelessWidget {
  final String linkToGo;
  final String link;
  final String text;
  final String desc;
  final Function() onRateUp;
  final Function() onRateDown;

  const SearchResultComponent({
    Key? key,
    required this.linkToGo,
    required this.link,
    required this.text,
    required this.desc,
    required this.onRateUp,
    required this.onRateDown,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.start,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: EdgeInsets.only(bottom: 8.0),
          child: LinkText(
            link: link,
            text: text,
            onTap: () async {
              Navigator.pushNamed(context, "/document?documentDir=$linkToGo");
            },
          ),
        ),
        Text(
          desc,
          style: TextStyle(fontSize: 14, color: Color(0xFF4d5156)),
        ),
        Row(
          children: [
            IconButton(onPressed: onRateUp, icon: Icon(Icons.thumb_up)),
            IconButton(onPressed: onRateDown, icon: Icon(Icons.thumb_down)),
          ],
        )
      ],
    );
  }
}

class LinkText extends StatefulWidget {
  final String link;
  final String text;
  final Function()? onTap;
  final TextStyle? textStyle;

  const LinkText(
      {Key? key,
      required this.link,
      required this.text,
      this.onTap,
      this.textStyle})
      : super(key: key);

  @override
  _LinkTextState createState() => _LinkTextState();
}

class _LinkTextState extends State<LinkText> {
  bool _showUnderLine = false;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      splashColor: Colors.transparent,
      highlightColor: Colors.transparent,
      hoverColor: Colors.transparent,
      onTap: widget.onTap,
      onHover: (hovering) {
        setState(() {
          _showUnderLine = hovering;
        });
      },
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            widget.link,
            style: TextStyle(fontSize: 14, color: Color(0xFF202124)),
          ),
          SizedBox(height: 5),
          Text(
            widget.text,
            style: widget.textStyle != null
                ? widget.textStyle?.copyWith(
                    decoration: _showUnderLine
                        ? TextDecoration.underline
                        : TextDecoration.none)
                : TextStyle(
                    color: Color(0xFF1a0dab),
                    fontWeight: FontWeight.w400,
                    fontSize: 20,
                    decoration: _showUnderLine
                        ? TextDecoration.underline
                        : TextDecoration.none,
                  ),
          ),
        ],
      ),
    );
  }
}
