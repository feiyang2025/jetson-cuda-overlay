/*
 * icu66shim.c — ABI shim for libQMapLibre (built against ICU 66) on systems
 * with newer ICU (74). QMapLibre references 13 ICU symbols from libicui18n /
 * libicuuc with the _66 suffix; ICU 74 only exports _74. This shim exports
 * the _66 names and forwards to the _74 implementations.
 *
 * Build: gcc -shared -fPIC -O2 -o libicu66shim.so icu66shim.c -licui18n -licuuc
 * Install: /usr/local/lib + ldconfig (or add to LIBPATH/RPATH of the linker)
 *
 * AGX Orin, 2026-09-15 (cpv9-cuda-main ui link fix).
 */
#include <stdint.h>

/* ------------------- forward decls of ICU 74 symbols ------------------- */
typedef void *UBiDi;
typedef int32_t UErrorCode;
typedef int32_t UVersionInfo[4];

extern UBiDi ubidi_open_74(void);
extern void ubidi_close_74(UBiDi *pBiDi);
extern void ubidi_setPara_74(UBiDi *pBiDi, const uint16_t *text, int32_t length,
                             uint8_t paraLevel, uint8_t *embeddingLevels,
                             UErrorCode *pErrorCode);
extern void ubidi_setLine_74(const UBiDi *pParaBiDi, int32_t start, int32_t limit,
                             UBiDi *pLineBiDi, UErrorCode *pErrorCode);
extern int32_t ubidi_countParagraphs_74(UBiDi *pBiDi);
extern void ubidi_getParagraphByIndex_74(const UBiDi *pBiDi, int32_t paraIndex,
                                         int32_t *pParaStart, int32_t *pParaLimit,
                                         uint8_t *pParaDirection, UErrorCode *pErrorCode);
extern int32_t ubidi_countRuns_74(UBiDi *pBiDi, UErrorCode *pErrorCode);
extern void ubidi_getVisualRun_74(UBiDi *pBiDi, int32_t runIndex,
                                  int32_t *pLogicalStart, int32_t *pLength,
                                  UErrorCode *pErrorCode);
extern int32_t ubidi_getProcessedLength_74(const UBiDi *pBiDi);
extern int32_t ubidi_writeReordered_74(UBiDi *pBiDi, uint16_t *dest, int32_t destSize,
                                       uint16_t options, UErrorCode *pErrorCode);
extern int32_t ubidi_writeReverse_74(const uint16_t *src, int32_t srcLength,
                                     uint16_t *dest, int32_t destSize,
                                     uint16_t options, UErrorCode *pErrorCode);
extern const char *u_errorName_74(UErrorCode code);
extern int32_t u_shapeArabic_74(const uint16_t *source, int32_t sourceLength,
                                uint16_t *dest, int32_t destSize,
                                uint32_t options, UErrorCode *pErrorCode);

/* ------------------------- the _66 forwarding stubs ------------------------- */
UBiDi ubidi_open_66(void) { return ubidi_open_74(); }
void ubidi_close_66(UBiDi *pBiDi) { ubidi_close_74(pBiDi); }
void ubidi_setPara_66(UBiDi *pBiDi, const uint16_t *text, int32_t length,
                      uint8_t paraLevel, uint8_t *embeddingLevels,
                      UErrorCode *pErrorCode) {
  ubidi_setPara_74(pBiDi, text, length, paraLevel, embeddingLevels, pErrorCode);
}
void ubidi_setLine_66(const UBiDi *pParaBiDi, int32_t start, int32_t limit,
                      UBiDi *pLineBiDi, UErrorCode *pErrorCode) {
  ubidi_setLine_74(pParaBiDi, start, limit, pLineBiDi, pErrorCode);
}
int32_t ubidi_countParagraphs_66(UBiDi *pBiDi) { return ubidi_countParagraphs_74(pBiDi); }
void ubidi_getParagraphByIndex_66(const UBiDi *pBiDi, int32_t paraIndex,
                                  int32_t *pParaStart, int32_t *pParaLimit,
                                  uint8_t *pParaDirection, UErrorCode *pErrorCode) {
  ubidi_getParagraphByIndex_74(pBiDi, paraIndex, pParaStart, pParaLimit,
                               pParaDirection, pErrorCode);
}
int32_t ubidi_countRuns_66(UBiDi *pBiDi, UErrorCode *pErrorCode) {
  return ubidi_countRuns_74(pBiDi, pErrorCode);
}
void ubidi_getVisualRun_66(UBiDi *pBiDi, int32_t runIndex,
                           int32_t *pLogicalStart, int32_t *pLength,
                           UErrorCode *pErrorCode) {
  ubidi_getVisualRun_74(pBiDi, runIndex, pLogicalStart, pLength, pErrorCode);
}
int32_t ubidi_getProcessedLength_66(const UBiDi *pBiDi) {
  return ubidi_getProcessedLength_74(pBiDi);
}
int32_t ubidi_writeReordered_66(UBiDi *pBiDi, uint16_t *dest, int32_t destSize,
                                uint16_t options, UErrorCode *pErrorCode) {
  return ubidi_writeReordered_74(pBiDi, dest, destSize, options, pErrorCode);
}
int32_t ubidi_writeReverse_66(const uint16_t *src, int32_t srcLength,
                              uint16_t *dest, int32_t destSize,
                              uint16_t options, UErrorCode *pErrorCode) {
  return ubidi_writeReverse_74(src, srcLength, dest, destSize, options, pErrorCode);
}
const char *u_errorName_66(UErrorCode code) { return u_errorName_74(code); }
int32_t u_shapeArabic_66(const uint16_t *source, int32_t sourceLength,
                         uint16_t *dest, int32_t destSize,
                         uint32_t options, UErrorCode *pErrorCode) {
  return u_shapeArabic_74(source, sourceLength, dest, destSize, options, pErrorCode);
}