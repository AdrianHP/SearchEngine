import 'package:shared_preferences/shared_preferences.dart';

abstract class IApiConfigurationService {
  Future<String> getHost();
  Future<int> getPort();
  Future<String> getScheme();
  Future<void> setHost(String host);
  Future<void> setPort(int port);
  Future<void> setScheme(String scheme);
  Future<Uri> getUrl(String path, {Map<String, String>? queryParams});
}

class ApiConfigurationService implements IApiConfigurationService {
  static const _hostKey = "host_key";
  static const _hostDefault = "127.0.0.1";

  static const _portKey = "port_key";
  static const _portDefault = "8000";

  static const _schemeKey = "scheme_key";
  static const _schemeDefault = "http";

  Future<String> _getAndSaveKey(String key, String defaultValue) async {
    final sharedPreferences = await SharedPreferences.getInstance();
    final value = sharedPreferences.getString(key);
    if (value == null) {
      await sharedPreferences.setString(key, defaultValue);
    }
    return value ?? defaultValue;
  }

  @override
  Future<String> getHost() {
    final host = _getAndSaveKey(_hostKey, _hostDefault);
    return host;
  }

  @override
  Future<int> getPort() async {
    final port = await _getAndSaveKey(_portKey, _portDefault);
    return int.parse(port);
  }

  @override
  Future<void> setHost(String host) async {
    final sharedPreferences = await SharedPreferences.getInstance();
    await sharedPreferences.setString(_hostKey, host);
  }

  @override
  Future<void> setPort(int port) async {
    final sharedPreferences = await SharedPreferences.getInstance();
    sharedPreferences.setString(_portKey, port.toString());
  }

  @override
  Future<String> getScheme() async {
    final scheme = await _getAndSaveKey(_schemeKey, _schemeDefault);
    return scheme;
  }

  @override
  Future<void> setScheme(String scheme) async {
    final sharedPreferences = await SharedPreferences.getInstance();
    await sharedPreferences.setString(_schemeKey, scheme);
  }

  @override
  Future<Uri> getUrl(String path, {Map<String, String>? queryParams}) async {
    final host = await getHost();
    final port = await getPort();
    final scheme = await getScheme();
    return Uri(
        scheme: scheme,
        host: host,
        port: port,
        path: path,
        queryParameters: queryParams);
  }
}
