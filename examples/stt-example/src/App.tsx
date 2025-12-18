import { useState, useEffect, useRef } from "react";
import {
  Box,
  Button,
  Typography,
  Paper,
  Stack,
  Alert,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
  List,
  ListItem,
  ListItemText,
  Chip,
  TextField,
  FormControlLabel,
  Switch,
  Container,
} from "@mui/material";
import {
  startListening,
  stopListening,
  isAvailable as sttIsAvailable,
  getSupportedLanguages,
  checkPermission as sttCheckPermission,
  requestPermission as sttRequestPermission,
  onResult,
  onStateChange,
  onError,
  type SttError,
} from "tauri-plugin-stt-api";
import {
  MdMic,
  MdStop,
  MdCheckCircle,
  MdError,
  MdRefresh,
} from "react-icons/md";

export default function App() {
  // STT state
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState<string>("");
  const [partialTranscript, setPartialTranscript] = useState<string>("");
  const [language, setLanguage] = useState<string>("en-US");
  const [availableLanguages, setAvailableLanguages] = useState<
    Array<{ code: string; name: string; installed?: boolean }>
  >([]);
  const [interimResults, setInterimResults] = useState(true);
  const [continuous, setContinuous] = useState(true);
  const [maxAlternatives, setMaxAlternatives] = useState(1);

  // UI state
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);
  const [availabilityReason, setAvailabilityReason] = useState<string | null>(
    null
  );
  const [permission, setPermission] = useState<{
    microphone: string;
    speechRecognition: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Download progress state
  const [downloadProgress, setDownloadProgress] = useState<{
    status: string;
    model: string;
    progress: number;
  } | null>(null);

  // Results history
  const [results, setResults] = useState<
    Array<{
      text: string;
      isFinal: boolean;
      confidence?: number;
      timestamp: Date;
    }>
  >([]);

  const resultsEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new results arrive
  useEffect(() => {
    resultsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [results]);

  // Check availability and load languages on mount
  useEffect(() => {
    checkAvailability();
  }, []);

  // Load languages and check permissions only when STT is available
  useEffect(() => {
    if (isAvailable === true) {
      loadLanguages();
      checkPerm();
    }
  }, [isAvailable]);

  // Listen for download progress events
  useEffect(() => {
    let unlisten: (() => void) | undefined;

    const setupListener = async () => {
      const { listen } = await import("@tauri-apps/api/event");
      unlisten = await listen<{
        status: string;
        model: string;
        progress: number;
      }>("stt://download-progress", event => {
        setDownloadProgress(event.payload);
        if (event.payload.status === "complete") {
          setSuccess(`Model ${event.payload.model} downloaded successfully!`);
          setTimeout(() => {
            setDownloadProgress(null);
            setSuccess(null);
            loadLanguages(); // Refresh languages to show installed status
          }, 2000);
        }
      });
    };

    setupListener();

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  const checkAvailability = async () => {
    try {
      const result = await sttIsAvailable();
      setIsAvailable(result.available);
      setAvailabilityReason(result.reason || null);

      if (result.available) {
        setSuccess("STT is available!");
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError(result.reason || "STT is not available");
      }
    } catch (err) {
      const errorMsg = String(err);
      if (errorMsg.includes("Plugin not found")) {
        setError(
          "STT plugin is not enabled. Build with --features stt to enable it."
        );
        setAvailabilityReason("Plugin not compiled with STT support");
      } else {
        setError(`Failed to check availability: ${err}`);
      }
      setIsAvailable(false);
    }
  };

  const checkPerm = async () => {
    try {
      const perm = await sttCheckPermission();
      setPermission({
        microphone: perm.microphone,
        speechRecognition: perm.speechRecognition,
      });
    } catch (err) {
      // Silently fail if plugin not available - will be caught by checkAvailability
      if (!String(err).includes("Plugin not found")) {
        setError(`Failed to check permission: ${err}`);
      }
    }
  };

  const handleRequestPermission = async () => {
    try {
      const perm = await sttRequestPermission();
      setPermission({
        microphone: perm.microphone,
        speechRecognition: perm.speechRecognition,
      });

      if (perm.microphone === "granted") {
        setSuccess("Permission granted!");
        setTimeout(() => setSuccess(null), 3000);
      }
    } catch (err) {
      setError(`Failed to request permission: ${err}`);
    }
  };

  const loadLanguages = async () => {
    try {
      const result = await getSupportedLanguages();
      setAvailableLanguages(result.languages);
    } catch (err) {
      // Only show error if it's not about plugin not being available
      if (!String(err).includes("Plugin not found")) {
        setError(`Failed to load languages: ${err}`);
      }
    }
  };

  const handleStartListening = async () => {
    setError(null);
    setLoading(true);
    setPartialTranscript("");

    try {
      // Register listeners for STT events using the guest SDK helpers
      const resultListener = await onResult(result => {
        const { transcript: text, isFinal, confidence } = result;

        if (isFinal) {
          // Apenas resultados finais são adicionados ao histórico e transcrição
          setTranscript(prev => prev + (prev ? " " : "") + text);
          setPartialTranscript("");
          setResults(prev => [
            ...prev,
            {
              text,
              isFinal: true,
              confidence,
              timestamp: new Date(),
            },
          ]);
        } else {
          // Resultados parciais apenas atualizam o display, não entram no histórico
          setPartialTranscript(text);
        }
      });

      const stateListener = await onStateChange(() => {
        // State changes tracked silently
      });

      // Listen for errors
      const errorListener = await onError((errorEvent: SttError) => {
        console.error("STT error:", errorEvent);
        setError(
          `STT Error: ${errorEvent.message ?? errorEvent.code ?? String(errorEvent)}`
        );
        setLoading(false);
      });

      // Store listeners for cleanup on stop
      (
        window as unknown as {
          sttListeners?: ((() => void) | { unregister?: () => void })[];
        }
      ).sttListeners = [resultListener, stateListener, errorListener];

      // Start listening
      await startListening({
        language,
        interimResults: interimResults,
        continuous,
      });

      setIsListening(true);
      setSuccess("Started listening");
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      const errorMessage = String(err);
      if (errorMessage.includes("permission")) {
        setError(
          `Permissão necessária: ${err}. Por favor, conceda a permissão e tente novamente.`
        );
      } else {
        setError(`Failed to start listening: ${err}`);
      }
      setIsListening(false);
    } finally {
      setLoading(false);
    }
  };

  const handleStopListening = async () => {
    setLoading(true);
    try {
      await stopListening();

      // Clean up listeners
      const listeners = (
        window as unknown as {
          sttListeners?: (() => void)[];
        }
      ).sttListeners;
      if (listeners) {
        for (const listener of listeners) {
          if (typeof listener === "function") {
            listener();
          } else if (
            listener &&
            typeof (listener as { unregister?: () => void }).unregister ===
              "function"
          ) {
            (listener as { unregister: () => void }).unregister();
          }
        }
        (window as unknown as { sttListeners?: unknown[] }).sttListeners =
          undefined;
      }

      setIsListening(false);
      setPartialTranscript("");
      setSuccess("Stopped listening");
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      setError(`Failed to stop listening: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClearResults = () => {
    setTranscript("");
    setPartialTranscript("");
    setResults([]);
    setSuccess("Cleared results");
    setTimeout(() => setSuccess(null), 2000);
  };

  const getPermissionColor = (status: string) => {
    switch (status) {
      case "granted":
        return "success";
      case "denied":
        return "error";
      default:
        return "warning";
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: { xs: 2, sm: 3, md: 4 } }}>
      <Paper
        elevation={0}
        sx={{
          p: { xs: 2, sm: 3 },
          mb: { xs: 2, sm: 3 },
          borderRadius: 2,
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          color: "white",
        }}
      >
        <Typography
          variant="h4"
          component="h1"
          sx={{
            fontSize: { xs: "1.5rem", sm: "2rem", md: "2.125rem" },
            fontWeight: 700,
            mb: 1,
          }}
        >
          🎤 Speech-to-Text Example
        </Typography>
        <Typography
          variant="body2"
          sx={{
            fontSize: { xs: "0.875rem", sm: "1rem" },
            opacity: 0.9,
            display: { xs: "none", sm: "block" },
          }}
        >
          Test the native Speech-to-Text plugin functionality
        </Typography>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert
          severity="success"
          sx={{ mb: 2 }}
          onClose={() => setSuccess(null)}
        >
          {success}
        </Alert>
      )}

      <Stack spacing={{ xs: 2, sm: 3 }}>
        {downloadProgress && (
          <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
            <Stack spacing={{ xs: 1.5, sm: 2 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <CircularProgress size={24} />
                <Box sx={{ flex: 1 }}>
                  <Typography
                    sx={{
                      color: "white",
                      fontSize: { xs: "0.875rem", sm: "1rem" },
                    }}
                  >
                    {downloadProgress.status === "downloading"
                      ? `Downloading ${downloadProgress.model}...`
                      : downloadProgress.status === "extracting"
                        ? `Extracting ${downloadProgress.model}...`
                        : `${downloadProgress.model} ready!`}
                  </Typography>
                  <Box
                    sx={{
                      mt: 1,
                      height: 4,
                      bgcolor: "rgba(255,255,255,0.2)",
                      borderRadius: 2,
                    }}
                  >
                    <Box
                      sx={{
                        width: `${downloadProgress.progress}%`,
                        height: "100%",
                        bgcolor: "#22c55e",
                        borderRadius: 2,
                        transition: "width 0.3s",
                      }}
                    />
                  </Box>
                </Box>
              </Box>
            </Stack>
          </Paper>
        )}

        {/* Availability Status */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Stack spacing={{ xs: 1.5, sm: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              {isAvailable === null ? (
                <CircularProgress size={20} />
              ) : isAvailable ? (
                <MdCheckCircle color="#22c55e" size={24} />
              ) : (
                <MdError color="#ef4444" size={24} />
              )}
              <Typography
                variant="h6"
                sx={{ fontSize: { xs: "1rem", sm: "1.25rem" } }}
              >
                {isAvailable === null
                  ? "Checking availability..."
                  : isAvailable
                    ? "STT Available"
                    : "STT Not Available"}
              </Typography>
            </Box>

            {availabilityReason && (
              <Alert severity="info">{availabilityReason}</Alert>
            )}

            {isAvailable === false && (
              <Button
                variant="outlined"
                onClick={checkAvailability}
                startIcon={<MdRefresh />}
                sx={{ minHeight: { xs: 48, sm: 44 } }}
              >
                Recheck
              </Button>
            )}
          </Stack>
        </Paper>

        {/* Permission Status */}
        {permission && (
          <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
            <Typography
              variant="h6"
              sx={{
                mb: { xs: 1.5, sm: 2 },
                fontSize: { xs: "1rem", sm: "1.25rem" },
              }}
            >
              Permissions
            </Typography>
            <Stack spacing={1}>
              <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                <Chip
                  label={`Microphone: ${permission?.microphone || "Checking..."}`}
                  color={getPermissionColor(
                    permission?.microphone || "unknown"
                  )}
                  size="small"
                />
                <Chip
                  label={`Speech Recognition: ${permission?.speechRecognition || "Checking..."}`}
                  color={getPermissionColor(
                    permission?.speechRecognition || "unknown"
                  )}
                  size="small"
                />
              </Box>
              {permission && permission.microphone !== "granted" && (
                <Button
                  variant="contained"
                  size="small"
                  onClick={handleRequestPermission}
                  sx={{
                    alignSelf: "flex-start",
                    minHeight: { xs: 48, sm: 44 },
                  }}
                >
                  Request Permission
                </Button>
              )}
            </Stack>
          </Paper>
        )}

        {/* Configuration */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Typography
            variant="h6"
            sx={{
              mb: { xs: 1.5, sm: 2 },
              fontSize: { xs: "1rem", sm: "1.25rem" },
            }}
          >
            Configuration
          </Typography>
          <Stack spacing={2}>
            {/* Language Selection */}
            <Box>
              <Typography
                sx={{ mb: 1, fontSize: { xs: "0.875rem", sm: "1rem" } }}
              >
                Language (✓ = installed)
              </Typography>
              <ToggleButtonGroup
                value={language}
                exclusive
                onChange={(_, value) => value && setLanguage(value)}
                disabled={isListening || downloadProgress !== null}
                size="small"
                sx={{ flexWrap: "wrap" }}
              >
                {availableLanguages.map(lang => (
                  <ToggleButton
                    key={lang.code}
                    value={lang.code}
                    sx={{
                      color: lang.installed
                        ? "#22c55e"
                        : "rgba(255,255,255,0.6)",
                    }}
                  >
                    {lang.code} {lang.installed ? "✓" : "↓"}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
              {!availableLanguages.find(l => l.code === language)
                ?.installed && (
                <Typography
                  sx={{
                    color: "rgba(255,255,255,0.6)",
                    fontSize: { xs: "0.625rem", sm: "0.75rem" },
                    mt: 1,
                  }}
                >
                  Model will be downloaded automatically when you start
                  listening
                </Typography>
              )}
            </Box>

            {/* Options */}
            <Stack
              direction={{ xs: "column", sm: "row" }}
              spacing={{ xs: 1.5, sm: 2 }}
            >
              <FormControlLabel
                control={
                  <Switch
                    checked={interimResults}
                    onChange={e => setInterimResults(e.target.checked)}
                    disabled={isListening}
                  />
                }
                label="Interim Results"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={continuous}
                    onChange={e => setContinuous(e.target.checked)}
                    disabled={isListening}
                  />
                }
                label="Continuous"
              />
            </Stack>

            <TextField
              label="Max Alternatives"
              type="number"
              value={maxAlternatives}
              onChange={e => setMaxAlternatives(Number(e.target.value))}
              disabled={isListening}
              size="small"
              sx={{ maxWidth: 200 }}
            />
          </Stack>
        </Paper>

        {/* Controls */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={{ xs: 1.5, sm: 2 }}
            sx={{ justifyContent: "center" }}
          >
            {!isListening ? (
              <Button
                variant="contained"
                size="large"
                onClick={handleStartListening}
                disabled={loading || isAvailable === false}
                startIcon={loading ? <CircularProgress size={20} /> : <MdMic />}
                sx={{
                  bgcolor: "#ef4444",
                  "&:hover": { bgcolor: "#dc2626" },
                  minHeight: { xs: 48, sm: 44 },
                }}
              >
                Start Listening
              </Button>
            ) : (
              <Button
                variant="contained"
                size="large"
                onClick={handleStopListening}
                disabled={loading}
                startIcon={
                  loading ? <CircularProgress size={20} /> : <MdStop />
                }
                sx={{
                  bgcolor: "#dc2626",
                  "&:hover": { bgcolor: "#b91c1c" },
                  minHeight: { xs: 48, sm: 44 },
                }}
              >
                Stop
              </Button>
            )}

            <Button
              variant="outlined"
              onClick={handleClearResults}
              disabled={isListening}
              startIcon={<MdRefresh />}
              sx={{ minHeight: { xs: 48, sm: 44 } }}
            >
              Clear
            </Button>
          </Stack>
        </Paper>

        {/* Transcription Display */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Typography
            variant="h6"
            sx={{
              mb: { xs: 1.5, sm: 2 },
              fontSize: { xs: "1rem", sm: "1.25rem" },
            }}
          >
            Transcription
          </Typography>

          {isListening && (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                mb: { xs: 1.5, sm: 2 },
                color: "#ef4444",
              }}
            >
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  bgcolor: "#ef4444",
                  animation: "pulse 1s infinite",
                  "@keyframes pulse": {
                    "0%, 100%": { opacity: 1 },
                    "50%": { opacity: 0.5 },
                  },
                }}
              />
              <Typography
                variant="body2"
                sx={{ fontSize: { xs: "0.875rem", sm: "1rem" } }}
              >
                Listening...
              </Typography>
            </Box>
          )}

          <Box
            sx={{
              minHeight: { xs: 80, sm: 100 },
              p: { xs: 1.5, sm: 2 },
              borderRadius: 1,
              bgcolor: "action.hover",
              fontSize: { xs: "1rem", sm: "1.125rem" },
              lineHeight: 1.6,
            }}
          >
            {transcript}
            {partialTranscript && (
              <span style={{ opacity: 0.5 }}> {partialTranscript}</span>
            )}
            {!transcript && !partialTranscript && (
              <Typography color="text.secondary">
                Start listening to see transcription here...
              </Typography>
            )}
          </Box>
        </Paper>

        {results.length > 0 && (
          <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
            <Typography
              variant="h6"
              sx={{
                mb: { xs: 1.5, sm: 2 },
                fontSize: { xs: "1rem", sm: "1.25rem" },
              }}
            >
              Results History ({results.length})
            </Typography>
            <List
              sx={{
                maxHeight: { xs: 200, sm: 300 },
                overflow: "auto",
                bgcolor: "action.hover",
                borderRadius: 1,
              }}
            >
              {results.map((result, index) => (
                <ListItem key={index} divider>
                  <ListItemText
                    primary={result.text}
                    secondary={
                      <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                        <Chip
                          label={result.isFinal ? "Final" : "Partial"}
                          size="small"
                          color={result.isFinal ? "success" : "default"}
                        />
                        {result.confidence !== undefined && (
                          <Chip
                            label={`${(result.confidence * 100).toFixed(0)}%`}
                            size="small"
                          />
                        )}
                        <Chip
                          label={result.timestamp.toLocaleTimeString()}
                          size="small"
                        />
                      </Stack>
                    }
                    sx={{}}
                  />
                </ListItem>
              ))}
              <div ref={resultsEndRef} />
            </List>
          </Paper>
        )}
      </Stack>
    </Container>
  );
}
