/**
 * Tianjin Mahjong Strategy Engine
 *
 * Tianjin rules:
 * - 136 tiles: 万(1-9) 筒(1-9) 条(1-9) 东南西北 中发白
 * - 碰 allowed, 吃 NOT allowed
 * - No 放炮 — must self-draw (自摸)
 * - 混儿 (hun er) — wild card system
 * - Higher minimum fan (混儿吊 alone insufficient)
 *
 * Tile encoding:
 *   1m-9m = 万 (Characters)
 *   1p-9p = 筒 (Dots)
 *   1s-9s = 条 (Bamboo)
 *   E, S, W, N = 东南西北
 *   R, G, Wt = 中发白 (Red, Green, White)
 *
 * For internal computation, tiles are mapped to integers 0-33:
 *   0-8:   万 1-9
 *   9-17:  筒 1-9
 *   18-26: 条 1-9
 *   27: 东, 28: 南, 29: 西, 30: 北
 *   31: 中, 32: 发, 33: 白
 */

const TIANJIN_ENGINE = (() => {
  // Tile encoding maps
  const TILE_TO_ID = {
    '1m':0,'2m':1,'3m':2,'4m':3,'5m':4,'6m':5,'7m':6,'8m':7,'9m':8,
    '1p':9,'2p':10,'3p':11,'4p':12,'5p':13,'6p':14,'7p':15,'8p':16,'9p':17,
    '1s':18,'2s':19,'3s':20,'4s':21,'5s':22,'6s':23,'7s':24,'8s':25,'9s':26,
    'E':27, 'S':28, 'W':29, 'N':30, 'R':31, 'G':32, 'Wt':33,
  };

  const ID_TO_TILE = {};
  for (const [k, v] of Object.entries(TILE_TO_ID)) {
    ID_TO_TILE[v] = k;
  }

  const TILE_NAMES = [
    '1万','2万','3万','4万','5万','6万','7万','8万','9万',
    '1筒','2筒','3筒','4筒','5筒','6筒','7筒','8筒','9筒',
    '1条','2条','3条','4条','5条','6条','7条','8条','9条',
    '东','南','西','北','中','发','白',
  ];

  const NUM_TILES = 34;
  const MAX_PER_TILE = 4;

  // Suit ranges
  const SUIT_RANGES = {
    'm': [0, 8],   // 万
    'p': [9, 17],  // 筒
    's': [18, 26], // 条
    'z': [27, 33], // 字 (honors)
  };

  // Fan values for patterns
  const FAN_VALUES = {
    '混儿吊': 1,   // Hun er tiao — winning pair uses wild card
    '素的': 2,     // Su de — no wild cards used
    '没混儿': 2,   // Mei hun er — no wild cards in hand
    '捉伍儿': 2,   // Zhuo wu er — winning with a 5 tile
    '龙': 2,       // Long — 1-9 straight in one suit
    '本混儿龙': 3, // Ben hun er long — dragon with wild card as the actual tile
    '清一色': 3,   // Qing yi se — all one suit
    '七对': 2,     // Qi dui — seven pairs
    '杠开': 1,     // Gang kai — win on kong draw
    '自摸': 1,     // Zi mo — self-draw (standard in Tianjin)
  };

  // === Helper Functions ===

  function tileId(tileStr) {
    return TILE_TO_ID[tileStr];
  }

  function tileName(tileId) {
    return TILE_NAMES[tileId] || ID_TO_TILE[tileId] || '?';
  }

  function tileSuit(tileId) {
    if (tileId < 9) return 'm';
    if (tileId < 18) return 'p';
    if (tileId < 27) return 's';
    return 'z';
  }

  function tileNum(tileId) {
    if (tileId < 9) return tileId + 1;
    if (tileId < 18) return tileId - 8;
    if (tileId < 27) return tileId - 17;
    return 0; // honors
  }

  /**
   * Parse tile string to array of IDs
   */
  function parseTiles(str) {
    if (!str || !str.trim()) return [];
    return str.split(',').map(s => {
      const t = s.trim();
      if (TILE_TO_ID[t] !== undefined) return TILE_TO_ID[t];
      return -1;
    }).filter(id => id >= 0);
  }

  /**
   * Count occurrences of each tile
   */
  function countTiles(tileIds) {
    const counts = new Array(NUM_TILES).fill(0);
    for (const id of tileIds) counts[id]++;
    return counts;
  }

  /**
   * Get remaining count for each tile (4 - seen)
   */
  function getRemaining(handIds, discardIds, meldIds) {
    const seen = new Array(NUM_TILES).fill(0);
    for (const id of handIds) seen[id]++;
    for (const id of discardIds) seen[id]++;
    for (const id of meldIds) seen[id]++;
    return seen.map(s => MAX_PER_TILE - s);
  }

  /**
   * Check if a tile is a 混儿 (wild card).
   * The 混儿 is the tile AFTER the dora indicator in sequence.
   * E.g., if dora is 3万 (id 2), then 混儿 is 4万 (id 3).
   */
  function isHun(tileId, hunDoraId) {
    if (hunDoraId === null || hunDoraId === undefined) return false;
    // The wild card wraps: if dora is 9 of a suit, wild is 1 of same suit
    const suit = tileSuit(hunDoraId);
    if (tileSuit(tileId) !== suit) return false;
    if (suit === 'z') {
      // Honors don't wrap around typically, but handle edge case
      if (hunDoraId === 33) return tileId === 27; // 白 → 东
      return tileId === hunDoraId + 1;
    }
    const doraNum = tileNum(hunDoraId);
    const wildNum = (doraNum % 9) + 1;
    return tileNum(tileId) === wildNum;
  }

  // === Pattern Detection ===

  /**
   * Check if hand contains 龙 (1-9 straight in one suit).
   * With wild cards, gaps can be filled.
   */
  function hasDragon(handCounts, hunTileIds) {
    for (const suit of ['m', 'p', 's']) {
      const [start, end] = SUIT_RANGES[suit];
      let wildsNeeded = 0;
      let hasEnough = true;

      for (let id = start; id <= end; id++) {
        const available = handCounts[id] + (hunTileIds && hunTileIds.includes(id) ? handCounts[id] : 0);
        // Actually, we need each number 1-9 present
        const count = handCounts[id];
        if (count === 0) {
          // Need a wild card to fill
          wildsNeeded++;
        }
      }

      const wildCount = hunTileIds
        ? hunTileIds.reduce((sum, hid) => sum + handCounts[hid], 0)
        : 0;

      if (wildsNeeded <= wildCount) return suit;
    }
    return null;
  }

  /**
   * Check if hand is 清一色 (all one suit, ignoring honors).
   * With wild cards, wilds can substitute for any suit.
   */
  function isQingYiSe(handCounts, hunTileIds) {
    const suits = new Set();
    const wildSet = new Set(hunTileIds || []);

    for (let id = 0; id < NUM_TILES; id++) {
      if (handCounts[id] > 0 && !wildSet.has(id)) {
        suits.add(tileSuit(id));
      }
    }

    return suits.size === 1;
  }

  /**
   * Check if hand has 混儿吊 (a pair formed by a wild card being used as the winning pair).
   * This means the hand completes with a wild card serving as the pair/eye.
   */
  function isHunErDiao(handCounts, hunTileIds, pairTile) {
    if (!hunTileIds || hunTileIds.length === 0) return false;
    return hunTileIds.includes(pairTile);
  }

  /**
   * Check if hand has 捉伍儿 (winning with a 5 tile as the pair).
   */
  function isZhuoWuEr(pairTile) {
    // 5万=4, 5筒=13, 5条=22
    return pairTile === 4 || pairTile === 13 || pairTile === 22;
  }

  /**
   * Check if hand has 素的 / 没混儿 (no wild cards used).
   */
  function isSuDe(handCounts, hunTileIds) {
    if (!hunTileIds || hunTileIds.length === 0) return true;
    return hunTileIds.every(hid => handCounts[hid] === 0);
  }

  /**
   * Detect all active patterns and compute total fan.
   * Returns { patterns: string[], totalFan: number }
   */
  function detectPatterns(handCounts, hunTileIds, pairTile) {
    const patterns = [];
    let totalFan = 0;

    // 混儿吊
    if (isHunErDiao(handCounts, hunTileIds, pairTile)) {
      patterns.push('混儿吊');
      totalFan += FAN_VALUES['混儿吊'];
    }

    // 素的
    if (isSuDe(handCounts, hunTileIds)) {
      patterns.push('素的');
      totalFan += FAN_VALUES['素的'];
    }

    // 捉伍儿
    if (isZhuoWuEr(pairTile)) {
      patterns.push('捉伍儿');
      totalFan += FAN_VALUES['捉伍儿'];
    }

    // 龙
    const dragon = hasDragon(handCounts, hunTileIds);
    if (dragon) {
      const hasWildInDragon = hunTileIds && hunTileIds.some(hid =>
        tileSuit(hid) === dragon
      );
      if (hasWildInDragon) {
        patterns.push('本混儿龙');
        totalFan += FAN_VALUES['本混儿龙'];
      } else {
        patterns.push('龙');
        totalFan += FAN_VALUES['龙'];
      }
    }

    // 清一色
    if (isQingYiSe(handCounts, hunTileIds)) {
      patterns.push('清一色');
      totalFan += FAN_VALUES['清一色'];
    }

    // 七对 (checked separately in isComplete)
    // 自摸 (always true in Tianjin since no discard win)
    patterns.push('自摸');
    totalFan += FAN_VALUES['自摸'];

    return { patterns, totalFan };
  }

  // === Shanten Calculation ===

  /**
   * Check if a hand (14 tiles) is a complete winning hand.
   * Handles regular hands (4 melds + 1 pair) and seven pairs.
   * Supports wild cards (混儿).
   */
  function isComplete(counts, hunTileIds) {
    const total = sum(counts);
    if (total !== 14) return false;

    // Check seven pairs
    if (isSevenPairs(counts, hunTileIds)) return true;

    // Check regular hand: try each possible pair
    const wildCount = hunTileIds ? hunTileIds.reduce((s, hid) => s + counts[hid], 0) : 0;

    for (let i = 0; i < NUM_TILES; i++) {
      if (counts[i] >= 2) {
        const remaining = [...counts];
        remaining[i] -= 2;
        if (canFormMelds(remaining, wildCount, hunTileIds)) return true;
      }
    }

    // Try wild card as pair
    if (wildCount >= 2 && hunTileIds) {
      const remaining = [...counts];
      // Remove 2 wild cards
      let toRemove = 2;
      for (const hid of hunTileIds) {
        const r = Math.min(toRemove, remaining[hid]);
        remaining[hid] -= r;
        toRemove -= r;
        if (toRemove === 0) break;
      }
      // Wild cards can also stand in for the pair
    }

    // Try pair formed by 1 wild + 1 regular tile
    if (wildCount >= 1 && hunTileIds) {
      for (let i = 0; i < NUM_TILES; i++) {
        if (counts[i] >= 1 && !hunTileIds.includes(i)) {
          const remaining = [...counts];
          remaining[i] -= 1;
          // Remove 1 wild
          for (const hid of hunTileIds) {
            if (remaining[hid] > 0) {
              remaining[hid]--;
              break;
            }
          }
          if (canFormMelds(remaining, wildCount - 1, hunTileIds)) return true;
        }
      }
    }

    return false;
  }

  function isSevenPairs(counts, hunTileIds) {
    let pairs = 0;
    let wildsAvailable = hunTileIds
      ? hunTileIds.reduce((s, hid) => s + counts[hid], 0)
      : 0;

    const remaining = [...counts];
    if (hunTileIds) {
      for (const hid of hunTileIds) remaining[hid] = 0;
    }

    for (let i = 0; i < NUM_TILES; i++) {
      if (remaining[i] >= 2) {
        pairs++;
        remaining[i] -= 2;
        if (remaining[i] >= 2) {
          pairs++;
          remaining[i] -= 2;
        }
      }
    }

    // Use wilds to complete remaining singles into pairs
    const singles = sum(remaining);
    const wildPairsNeeded = Math.ceil(singles / 2);
    if (wildPairsNeeded <= Math.floor(wildsAvailable / 2)) {
      pairs += (7 - pairs);
    }

    return pairs >= 7;
  }

  function canFormMelds(counts, wildCount, hunTileIds) {
    if (sum(counts) === 0) return true;

    // Try triplet of same tile
    for (let i = 0; i < NUM_TILES; i++) {
      if (counts[i] >= 3) {
        counts[i] -= 3;
        if (canFormMelds(counts, wildCount, hunTileIds)) {
          counts[i] += 3;
          return true;
        }
        counts[i] += 3;
      }
    }

    // Try triplet with wilds
    if (wildCount > 0) {
      for (let i = 0; i < NUM_TILES; i++) {
        const needed = 3 - counts[i];
        if (needed > 0 && needed <= wildCount && counts[i] > 0) {
          const orig = counts[i];
          counts[i] = 0;
          if (canFormMelds(counts, wildCount - needed, hunTileIds)) {
            counts[i] = orig;
            return true;
          }
          counts[i] = orig;
        }
      }
    }

    // Try sequence (only suited tiles, not honors)
    for (let suit of ['m', 'p', 's']) {
      const [start, end] = SUIT_RANGES[suit];
      for (let i = start; i <= end - 2; i++) {
        if (counts[i] > 0 && counts[i + 1] > 0 && counts[i + 2] > 0) {
          counts[i]--; counts[i + 1]--; counts[i + 2]--;
          if (canFormMelds(counts, wildCount, hunTileIds)) {
            counts[i]++; counts[i + 1]++; counts[i + 2]++;
            return true;
          }
          counts[i]++; counts[i + 1]++; counts[i + 2]++;
        }
      }
    }

    return false;
  }

  function sum(arr) {
    return arr.reduce((a, b) => a + b, 0);
  }

  /**
   * Calculate shanten (向听数) — distance from tenpai.
   * -1 = complete hand (can win)
   *  0 = tenpai (waiting)
   *  1+ = steps away from tenpai
   */
  function calcShanten(handIds, hunDoraId, remainingCounts) {
    const counts = countTiles(handIds);
    const total = handIds.length;
    const hunTileIds = findHunTileIds(hunDoraId);

    if (total === 14 && isComplete(counts, hunTileIds)) return -1;

    if (total === 13) {
      // Check if any tile would complete the hand
      const rem = remainingCounts || new Array(NUM_TILES).fill(4);
      for (let i = 0; i < NUM_TILES; i++) {
        if (rem[i] <= 0) continue;
        const testCounts = [...counts];
        testCounts[i]++;
        if (isComplete(testCounts, hunTileIds)) return 0;
      }
    }

    // Heuristic shanten calculation (adapted from Mortal/Sichuan approach)
    let best = 8;
    const wildCount = hunTileIds
      ? hunTileIds.reduce((s, hid) => s + counts[hid], 0)
      : 0;

    // Try each possible pair (雀头)
    for (let i = 0; i < NUM_TILES; i++) {
      if (counts[i] >= 2) {
        const r = [...counts];
        r[i] -= 2;
        const [mentsu, taatsu] = countMentsuTaatsu(r, wildCount);
        best = Math.min(best, 4 - mentsu - Math.min(taatsu, 4 - mentsu));
      }
    }

    // Try without explicit pair
    const [mentsu, taatsu] = countMentsuTaatsu([...counts], wildCount);
    best = Math.min(best, 5 - mentsu - Math.min(taatsu, 5 - mentsu));

    // Seven pairs shanten
    let pairs = 0;
    let wildsForPairs = wildCount;
    const tempCounts = [...counts];
    if (hunTileIds) {
      for (const hid of hunTileIds) tempCounts[hid] = 0;
    }
    for (let i = 0; i < NUM_TILES; i++) {
      pairs += Math.floor(tempCounts[i] / 2);
    }
    pairs += Math.floor(wildsForPairs / 2);
    best = Math.min(best, 6 - pairs);

    // Correction: if heuristic says tenpai but no actual waits found
    if (best === 0 && total === 13) {
      const rem = remainingCounts || new Array(NUM_TILES).fill(4);
      let found = false;
      for (let i = 0; i < NUM_TILES; i++) {
        if (rem[i] > 0) {
          const tc = [...counts];
          tc[i]++;
          if (isComplete(tc, hunTileIds)) { found = true; break; }
        }
      }
      if (!found) best = 1;
    }

    return Math.max(best, -1);
  }

  function countMentsuTaatsu(counts, wildCount) {
    let mentsu = 0, taatsu = 0;

    // Count triplets
    for (let i = 0; i < NUM_TILES; i++) {
      while (counts[i] >= 3) {
        counts[i] -= 3;
        mentsu++;
      }
    }

    // Count sequences in each suit
    for (const suit of ['m', 'p', 's']) {
      const [start, end] = SUIT_RANGES[suit];
      for (let i = start; i <= end - 2; i++) {
        while (counts[i] > 0 && counts[i + 1] > 0 && counts[i + 2] > 0) {
          counts[i]--; counts[i + 1]--; counts[i + 2]--;
          mentsu++;
        }
      }
    }

    // Count taatsu (proto-melds)
    // Pairs as potential taatsu (can become triplet or pair)
    for (let i = 0; i < NUM_TILES; i++) {
      if (counts[i] >= 2) {
        counts[i] -= 2;
        taatsu++;
      }
    }

    // Two-sided waits (ryanmen)
    for (const suit of ['m', 'p', 's']) {
      const [start, end] = SUIT_RANGES[suit];
      for (let i = start; i <= end - 1; i++) {
        if (counts[i] > 0 && counts[i + 1] > 0) {
          taatsu++;
          counts[i]--; counts[i + 1]--;
        }
      }
    }

    // Gapped waits (kanchan)
    for (const suit of ['m', 'p', 's']) {
      const [start, end] = SUIT_RANGES[suit];
      for (let i = start; i <= end - 2; i++) {
        if (counts[i] > 0 && counts[i + 2] > 0) {
          taatsu++;
          counts[i]--; counts[i + 2]--;
        }
      }
    }

    return [mentsu, taatsu];
  }

  function findHunTileIds(hunDoraId) {
    if (hunDoraId === null || hunDoraId === undefined) return [];
    const results = [];
    for (let id = 0; id < NUM_TILES; id++) {
      if (isHun(id, hunDoraId)) results.push(id);
    }
    return results;
  }

  /**
   * Find all waiting tiles for a tenpai hand.
   * Returns [{ tileId, remaining }]
   */
  function findWaits(handIds, hunDoraId, remainingCounts) {
    const counts = countTiles(handIds);
    const rem = remainingCounts || new Array(NUM_TILES).fill(4);
    const hunTileIds = findHunTileIds(hunDoraId);
    const waits = [];

    for (let i = 0; i < NUM_TILES; i++) {
      if (rem[i] <= 0) continue;
      const tc = [...counts];
      tc[i]++;
      if (tc[i] <= MAX_PER_TILE && isComplete(tc, hunTileIds)) {
        waits.push({ tileId: i, remaining: rem[i] });
      }
    }

    return waits;
  }

  /**
   * Calculate safety score for discarding a tile.
   * Higher = safer. Considers: discards seen, remaining count, edge vs middle.
   */
  function safetyScore(tileId, discardCounts, remainingCounts) {
    let score = 50;
    score += (discardCounts[tileId] || 0) * 15;
    if (remainingCounts[tileId] === 0) return 100;

    const num = tileNum(tileId);
    if (tileSuit(tileId) !== 'z') {
      if (num === 1 || num === 9) score += 10; // Edge tiles safer
      if (num >= 4 && num <= 6) score -= 10;   // Middle tiles more dangerous
    }

    return Math.min(100, Math.max(0, score));
  }

  /**
   * Analyze all possible discards from a 14-tile hand.
   * Returns ranked discard options.
   */
  function analyzeDiscard(handIds, discardIds, meldIds, hunDoraId) {
    const handCounts = countTiles(handIds);
    const discardCounts = countTiles(discardIds);
    const remaining = getRemaining(handIds, discardIds, meldIds);
    const hunTileIds = findHunTileIds(hunDoraId);

    const results = [];

    // Try discarding each unique tile in hand
    const seen = new Set();
    for (const tileId of handIds) {
      if (seen.has(tileId)) continue;
      seen.add(tileId);

      const testHand = handIds.filter((_, idx) => {
        // Find first occurrence and remove it
        const firstIdx = handIds.indexOf(tileId);
        return idx !== firstIdx || handIds.indexOf(tileId) !== idx;
      });

      // Actually simpler: just remove one instance
      const testIds = [...handIds];
      const removeIdx = testIds.indexOf(tileId);
      testIds.splice(removeIdx, 1);

      const shanten = calcShanten(testIds, hunDoraId, remaining);
      let waitCount = 0;
      let waits = [];

      if (shanten === 0) {
        waits = findWaits(testIds, hunDoraId, remaining);
        waitCount = waits.reduce((s, w) => s + w.remaining, 0);
      }

      const safety = safetyScore(tileId, discardCounts, remaining);

      results.push({
        tileId,
        tileName: tileName(tileId),
        shanten,
        waitCount,
        waits,
        safety,
      });
    }

    // Sort: lowest shanten → most waits → safest
    results.sort((a, b) => {
      if (a.shanten !== b.shanten) return a.shanten - b.shanten;
      if (b.waitCount !== a.waitCount) return b.waitCount - a.waitCount;
      return b.safety - a.safety;
    });

    return results;
  }

  /**
   * Get the safety emoji for a score.
   */
  function safetyEmoji(score) {
    if (score >= 70) return '🟢';
    if (score >= 40) return '🟡';
    return '🔴';
  }

  // === Public API ===
  return {
    TILE_TO_ID,
    ID_TO_TILE,
    TILE_NAMES,
    NUM_TILES,

    tileId,
    tileName,
    tileSuit,
    tileNum,
    parseTiles,
    countTiles,
    getRemaining,

    isHun,
    findHunTileIds,

    calcShanten,
    findWaits,
    analyzeDiscard,
    safetyScore,
    safetyEmoji,

    isComplete,
    detectPatterns,
    hasDragon,
    isQingYiSe,

    FAN_VALUES,
    SUIT_RANGES,
  };
})();

// Export for Node.js / module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TIANJIN_ENGINE;
}
