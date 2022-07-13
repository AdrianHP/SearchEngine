import 'package:flutter/material.dart';
import 'package:google_clone/pages/document_page.dart';
import 'package:google_clone/pages/search_results.dart';
import 'package:google_clone/pages/settings_page.dart';
import 'package:google_clone/provider/document_screen_provider.dart';
import 'package:google_clone/provider/global_provider.dart';
import 'package:google_clone/provider/search_result_provider.dart';
import 'package:google_clone/provider/settings_screen_provider.dart';
import 'package:google_clone/services/api_configuration_service.dart';
import 'package:google_clone/services/api_service.dart';
import 'package:provider/provider.dart';

import 'pages/main_page.dart';
import 'provider/main_screen_provider.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (context) => MainScreenProvider()),
        ChangeNotifierProvider(create: (context) => SearchResultProvider()),
        ChangeNotifierProvider(create: (context) => SettingsScreenProvider()),
        ChangeNotifierProvider(create: (context) => GlobalProvider()),
        ChangeNotifierProvider(create: (context) => DocumentScreenProvider()),
        Provider(create: (context) => ApiConfigurationService()),
        // Provider(create: (context) => ApiService()),
      ],
      child: App(),
    ),
  );
}

class App extends StatefulWidget {
  const App({Key? key}) : super(key: key);

  @override
  _AppState createState() => _AppState();
}

class _AppState extends State<App> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: Provider.of<GlobalProvider>(context).appName,
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        fontFamily: 'Arial',
      ),
      initialRoute: '/',
      onGenerateRoute: (settings) {
        final settingsURI = Uri.parse(settings.name ?? "/");
        switch (settingsURI.path) {
          case '/':
            return MaterialPageRoute(
              builder: (context) => GoogleSearchPage(),
              settings: RouteSettings(name: settings.name),
            );
          case '/search':
            return MaterialPageRoute(
              builder: (context) => GoogleSearchResultPage(
                q: settingsURI.queryParameters['q'] ?? 'google',
                startIndex: settingsURI.queryParameters['start'] ?? "1",
              ),
              settings: RouteSettings(name: settings.name),
            );
          case '/settings':
            return MaterialPageRoute(
                builder: (context) => SettingsPage(),
                settings: RouteSettings(name: settings.name));
          case '/document':
            return MaterialPageRoute(
                builder: (context) => DocumentPage(
                  documentDir: settingsURI.queryParameters["documentDir"] ?? "",
                ),
                settings: RouteSettings(name: settings.name));
          default:
            return MaterialPageRoute(
              builder: (context) => GoogleSearchPage(),
              settings: RouteSettings(name: settings.name),
            );
        }
      },
    );
  }
}
