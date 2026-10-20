package com.webcamrecognition;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * Small local-only HTTP service for the Fruit Vision browser client.
 * Model samples are deliberately saved on the local machine, never uploaded.
 */
public final class FruitVisionServer {
    private static final Path PUBLIC = Path.of("src/main/resources/public").toAbsolutePath();
    private static final Path MODEL_FILE = Path.of("data", "produce-samples.tsv");
    private static final List<Sample> samples = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        loadSamples();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 8080), 0);
        server.createContext("/", FruitVisionServer::handle);
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
        System.out.println("Fruit Vision is running at http://localhost:8080");
        System.out.println("Press Ctrl+C to stop.");
    }

    private static void handle(HttpExchange exchange) throws IOException {
        exchange.getResponseHeaders().set("Permissions-Policy", "camera=(self)");
        exchange.getResponseHeaders().set("Cache-Control", "no-store");
        String path = exchange.getRequestURI().getPath();
        try {
            if (path.equals("/api/labels") && exchange.getRequestMethod().equals("GET")) {
                json(exchange, 200, "{\"labels\":" + labelsJson() + ",\"sampleCount\":" + samples.size() + "}");
            } else if (path.equals("/api/model") && exchange.getRequestMethod().equals("GET")) {
                json(exchange, 200, "{\"samples\":" + samplesJson() + "}");
            } else if (path.equals("/api/samples") && exchange.getRequestMethod().equals("POST")) {
                saveSample(exchange);
            } else if (path.equals("/api/samples") && exchange.getRequestMethod().equals("DELETE")) {
                deleteLabel(exchange);
            } else if (exchange.getRequestMethod().equals("GET")) {
                serveAsset(exchange, path.equals("/") ? "/index.html" : path);
            } else {
                json(exchange, 405, "{\"error\":\"Method not allowed\"}");
            }
        } catch (IllegalArgumentException e) {
            json(exchange, 400, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        } catch (Exception e) {
            e.printStackTrace();
            json(exchange, 500, "{\"error\":\"Could not process the request\"}");
        }
    }

    private static synchronized void saveSample(HttpExchange exchange) throws IOException {
        Map<String, String> form = form(exchange);
        String label = form.getOrDefault("label", "").trim().replaceAll("\\s+", " ");
        String feature = form.getOrDefault("feature", "");
        if (!label.matches("[\\p{L}\\p{N} -]{2,40}")) {
            throw new IllegalArgumentException("Use a 2–40 character label with letters, numbers, spaces, or hyphens.");
        }
        String[] values = feature.split(",");
        if (values.length != 27) throw new IllegalArgumentException("The camera feature is invalid.");
        for (String value : values) {
            double number = Double.parseDouble(value);
            if (!Double.isFinite(number) || number < 0 || number > 255) {
                throw new IllegalArgumentException("The camera feature is invalid.");
            }
        }
        samples.add(new Sample(label, feature));
        persistSamples();
        json(exchange, 201, "{\"ok\":true,\"sampleCount\":" + samples.size() + "}");
    }

    private static synchronized void deleteLabel(HttpExchange exchange) throws IOException {
        String label = form(exchange).getOrDefault("label", "");
        samples.removeIf(sample -> sample.label.equalsIgnoreCase(label));
        persistSamples();
        json(exchange, 200, "{\"ok\":true}");
    }

    private static Map<String, String> form(HttpExchange exchange) throws IOException {
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        return java.util.Arrays.stream(body.split("&"))
                .filter(pair -> pair.contains("="))
                .map(pair -> pair.split("=", 2))
                .collect(java.util.stream.Collectors.toMap(
                        pair -> decode(pair[0]), pair -> decode(pair[1]), (first, ignored) -> first));
    }

    private static String decode(String value) {
        return URLDecoder.decode(value, StandardCharsets.UTF_8);
    }

    private static void serveAsset(HttpExchange exchange, String requestPath) throws IOException {
        Path asset = PUBLIC.resolve(requestPath.substring(1)).normalize();
        if (!asset.startsWith(PUBLIC) || !Files.isRegularFile(asset)) {
            json(exchange, 404, "{\"error\":\"Not found\"}");
            return;
        }
        String type = asset.toString().endsWith(".js") ? "text/javascript; charset=utf-8"
                : asset.toString().endsWith(".css") ? "text/css; charset=utf-8" : "text/html; charset=utf-8";
        byte[] bytes = Files.readAllBytes(asset);
        exchange.getResponseHeaders().set("Content-Type", type);
        exchange.sendResponseHeaders(200, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }

    private static synchronized void loadSamples() throws IOException {
        if (!Files.exists(MODEL_FILE)) return;
        for (String line : Files.readAllLines(MODEL_FILE, StandardCharsets.UTF_8)) {
            String[] parts = line.split("\\t", 2);
            if (parts.length == 2) samples.add(new Sample(new String(Base64.getDecoder().decode(parts[0]), StandardCharsets.UTF_8), parts[1]));
        }
    }

    private static synchronized void persistSamples() throws IOException {
        Files.createDirectories(MODEL_FILE.getParent());
        List<String> lines = samples.stream().map(s -> Base64.getEncoder().encodeToString(s.label.getBytes(StandardCharsets.UTF_8)) + "\t" + s.feature).toList();
        Files.write(MODEL_FILE, lines, StandardCharsets.UTF_8);
    }

    private static synchronized String labelsJson() {
        LinkedHashSet<String> labels = new LinkedHashSet<>();
        samples.stream().map(s -> s.label).sorted(Comparator.naturalOrder()).forEach(labels::add);
        return labels.stream().map(label -> "\"" + escape(label) + "\"").collect(java.util.stream.Collectors.joining(",", "[", "]"));
    }

    private static synchronized String samplesJson() {
        return samples.stream().map(sample -> "{\"label\":\"" + escape(sample.label) + "\",\"feature\":\"" + sample.feature + "\"}")
                .collect(java.util.stream.Collectors.joining(",", "[", "]"));
    }

    private static void json(HttpExchange exchange, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }

    private static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private record Sample(String label, String feature) {}
}
