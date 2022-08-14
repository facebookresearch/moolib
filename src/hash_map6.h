#pragma once

#include <cstddef>
#include <iterator>
#include <cstring>
#include <cassert>
#include <array>
#include <tuple>
#include <vector>

#if 0
#define likely(x) x
#define unlikely(x) x
//#define aligned(x, alignment) x
#else
#define likely(x) __builtin_expect(bool(x), 1)
#define unlikely(x) __builtin_expect(bool(x), 0)

//#define aligned(x, alignment) (std::remove_reference_t<decltype((x))>)__builtin_assume_aligned((x), alignment)
#endif

namespace moolib {

static constexpr int8_t none = -1;
template<typename T>
bool isNone(const T& v) {
  return v == (T)none;
}

template<typename Key, typename Value, typename Hash = std::hash<Key>>
struct HashMap {
private:
public:
  struct Item {
    Key key;
    Value value;
  };
  struct Group {
    static constexpr size_t nItems = std::min(std::max((size_t)60 / sizeof(Item), (size_t)1), (size_t)7);
    std::array<Item, nItems> arr;
    uint32_t ext;
    // uint16_t occupied;
    // int8_t distance;
    // int8_t maxDistance;
  };
  static_assert(sizeof(Group) <= 64);

  static constexpr size_t groupSize = std::max((size_t)64, sizeof(Group));

  size_t ksize = 0;
  size_t msize = 0;
  Group* groups = nullptr;
public:
  struct iterator {
  private:
  public:
    friend HashMap;
    HashMap* map;
    size_t gi;
    uint8_t vi;
    mutable std::aligned_storage_t<sizeof(std::pair<Key&, Value&>), alignof(std::pair<Key&, Value&>)> tmp;
  public:
    iterator() = default;
    iterator(const HashMap* map, size_t gi, uint8_t vi) : map(const_cast<HashMap*>(map)), gi(gi), vi(vi) {}
    iterator(const iterator& n) {
      map = n.map;
      gi = n.gi;
      vi = n.vi;
    }
    iterator& operator=(const iterator& n) {
      map = n.map;
      gi = n.gi;
      vi = n.vi;
      return *this;
    }

    using T = Value;

    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::bidirectional_iterator_tag;

    std::pair<Key&, Value&>& operator*() const noexcept {
      assert(gi < map->ksize);
      assert(vi < Group::nItems);
      Item& i = map->indexGroup(map->groups, gi)->arr[vi];
      new (&tmp)std::pair<Key&, Value&>(i.key, i.value);
      return (std::pair<Key&, Value&>&)tmp;
    }
    std::pair<Key&, Value&>* operator->() const noexcept {
      return &**this;
    }

    iterator& operator++() noexcept {
      do {
        if (vi == Group::nItems - 1) {
          vi = 0;
          ++gi;
          if (gi == map->ksize) {
            gi = none;
            vi = none;
            return *this;
          }
        } else {
          ++vi;
        }
        //printf("operator++, gi %d vi %d, ext %#x  (%#x)\n", gi, vi, map->indexGroup(map->groups, gi)->ext, ((map->indexGroup(map->groups, gi)->ext >> (4u * vi)) & 0xf));
      } while (((map->indexGroup(map->groups, gi)->ext >> (4u * vi)) & 0xf) == 0);
      //printf("return!\n");
      assert(gi < map->ksize);
      assert(vi < Group::nItems);
      return *this;
    }
    iterator operator++(int) noexcept {
      iterator r = *this;
      ++r;
      return r;
    }
    bool operator==(iterator n) const noexcept {
      return gi == n.gi && vi == n.vi;
    }
    bool operator!=(iterator n) const noexcept {
      return gi != n.gi || vi != n.vi;
    }
  };

  HashMap() = default;
  HashMap(const HashMap&) = delete;
  HashMap& operator=(const HashMap&) = delete;
  ~HashMap() {
    clear();
    deallocate(groups);
  }

  bool empty() const noexcept {
    return msize == 0;
  }

  void clear() noexcept {
    Group* end = indexGroup(groups, ksize);
    for (auto* i = groups; i != end; i = nextGroup(i)) {
      uint32_t distances = i->ext & 0xfffffff;
      i->ext = 0;
      while (distances) {
        uint8_t index = __builtin_ctzll(distances);
        index /= 4u;
        distances &= ~(0xfu << (4u * index));
        auto& v = i->arr[index];
        v.key.~Key();
        v.value.~Value();
      }
    }
    msize = 0;
    // while (i != e) {
    //   auto ni = std::next(i);
    //   erase(i);
    //   i = ni;
    // }
    assert(msize == 0);
    assert(begin() == end());
  }
  iterator begin() const noexcept {
    if (unlikely(!groups)) {
      return end();
    }
    size_t bs = ksize;
    auto* e = indexGroup(groups, bs);
    for (auto* i = groups; i != e; i = nextGroup(i)) {
      auto distances = i->ext & 0xfffffff;
      //printf("begin checking group %d with distances %#x\n", groupIndex(groups, i), distances);
      if (distances) {
        size_t index = __builtin_ctzll(distances) / 4u;
        return iterator(this, groupIndex(groups, i), index);
      }
    }
    return end();
  }
  iterator end() const noexcept {
    return iterator(this, none, none);
  }

  template<typename T>
  T* allocate(size_t n) {
    T* retval = (T*)std::aligned_alloc(std::max(alignof(T), (size_t)64), std::max(sizeof(T) * n, (size_t)64));
    std::memset(retval, 0xcc, std::max(sizeof(T) * n, (size_t)64));
    return retval;
  }
  template<typename T>
  void deallocate(T* ptr) {
    if (ptr) {
      std::free(ptr);
    }
  }

  template<typename KeyT>
  void erase(KeyT&& key) {
    auto i = find(std::forward<KeyT>(key));
    if (i != end()) {
      erase(i);
    }
  }

  template<typename KeyT>
  Value& operator[](KeyT&& index) {
    return try_emplace(std::forward<KeyT>(index)).first->second;
  }

  // struct MoveCacheLine {
  //   CacheLine dst;
  //   CacheLine src;
  //   MoveCacheLine(void* dstitem) : dst(dstitem) {}
  //   bool operator()(void* srcitem, size_t s) {
  //     src = CacheLine(srcitem);
  //     while (true) {
  //       if (srci->distance >= s) {
  //         if (dsti == dst.end()) {
  //           return false;
  //         }
  //         while (!isNone(dsti->distance)) {
  //           ++dsti;
  //           if (dsti == dst.end()) {
  //             return false;
  //           }
  //         }
  //         dsti->key = std::move(srci->key);
  //         dsti->value = std::move(srci->value);
  //         dsti->distance = 0;
  //         dsti->maxDistance = 0;
  //         srci->distance = none;
  //         ++dsti;
  //         if (dsti == dst.end()) {
  //           ++srci;
  //           return srci == src.end();
  //         }
  //       }
  //       ++srci;
  //       if (srci == src.end()) {
  //         return true;
  //       }
  //     }
  //   }
  // };

  Group* indexGroup(Group* base, size_t offset) const {
    return (Group*)((char*)base + offset * groupSize);
  }
  Group* nextGroup(Group* base) const {
    return indexGroup(base, 1);
  }
  size_t groupIndex(Group* base, Group* group) const {
    return (size_t((char*)group - (char*)base)) / groupSize;
  }

  void erase(iterator i) noexcept {
    assert(msize > 0);
    --msize;

    Group* groups = this->groups;
    Group* group = indexGroup(groups, i.gi);

    //printf("erase %d %d, size is now %d\n", i.gi, i.vi, msize);

    group->ext &= ~(0xf << (4 * i.vi));

    Item& item = group->arr[i.vi];
    item.key.~Key();
    item.value.~Value();

    // for (auto i = begin(); i != end(); ++i) {
    //   printf("  %d %d -> %d\n", i.gi, i.vi, i->first);
    // }

    // while (group->occupied == 0) {
    //   Group* nextGroup = nextGroup(group);
    //   if ((nextGroup->ext & 0xeeeeeeee) == 0) {
    //     break;
    //   }
    //   auto occupied = nextGroup->occupied;
    //   while (occupied) {
    //     size_t index = __builtin_ctzll(occupied);
    //     size_t distance = (occupied >> index) & 7;
    //     occupied &= ~(0xf << index);
    //     index /= 4;
    //     auto& v = nextGroup->arr[index];
    //     v.key.~Key();
    //     v.value.~Value();
    //   }
    //   nextGroup->occupied = 0;
    //   group = nextGroup;
    // }
  }

  template<typename KeyT, typename ValueT>
  iterator insert(KeyT&& key, ValueT&& value) {
    return try_emplace(std::forward<Key>(key), std::forward<Value>(value)).first;
  }

  void rehash(size_t newBs) noexcept {
    if (newBs & (newBs - 1)) {
      printf("bucket count is not a multiple of 2!\n");
      std::abort();
    }
    //printf("rehash %d %d\n", newBs, size());

    Group* oldGroups = groups;
    
    groups = (Group*)allocate<char>(groupSize * newBs);

    size_t bs = ksize;

    Group* g = groups;
    Group* ge = indexGroup(g, newBs);
    for (; g != ge; g = nextGroup(g)) {
      g->ext = 0;
    }
    size_t omsize = msize;
    msize = 0;
    ksize = newBs;

    if (oldGroups) {
      Group* end = indexGroup(oldGroups, bs);
      size_t n = 0;
      for (auto* i = oldGroups; i != end; i = nextGroup(i)) {
        uint32_t distances = i->ext & 0xfffffff;
        //printf("rehashing group %d with distances %#x\n", groupIndex(oldGroups, i), distances);
        while (distances) {
          uint8_t index = __builtin_ctzll(distances);
          index /= 4u;
          distances &= ~(0xfu << (4u * index));
          auto& v = i->arr[index];
          try_emplace(std::move(v.key), std::move(v.value));
          v.key.~Key();
          v.value.~Value();
          ++n;
        }
      }
      assert(n == msize);
      assert(n == omsize);
    }
    assert(msize == omsize);

    deallocate(oldGroups);
  }

#define forEachSet(distances, code) {\
auto s0 = distances & 0xf; \
auto s1 = distances & 0xf0; \
auto s2 = distances & 0xf00; \
auto s3 = distances & 0xf000; \
auto s4 = distances & 0xf0000; \
auto s5 = distances & 0xf00000; \
auto s6 = distances & 0xf000000; \
if (s0) {uint8_t index = 0; code}\
if (Group::nItems >= 2 && s1) {uint8_t index = 1; code}\
if (Group::nItems >= 3 && s2) {uint8_t index = 2; code}\
if (Group::nItems >= 4 && s3) {uint8_t index = 3; code}\
if (Group::nItems >= 5 && s4) {uint8_t index = 4; code}\
if (Group::nItems >= 6 && s5) {uint8_t index = 5; code}\
if (Group::nItems >= 7 && s6) {uint8_t index = 6; code}\
}

#define forEachSet_(distances, code) {\
if (distances) {uint32_t value__ = distances; uint8_t index = __builtin_ctz(distances); index = index / 4u; value__ >>= 4u * (index + 1); {code} \
if (Group::nItems >= 2 && value__) {size_t index__ = __builtin_ctzll(value__) / 4u + 1; index += index__; value__ >>= 4u * index__;{code}\
if (Group::nItems >= 3 && value__) {size_t index__ = __builtin_ctzll(value__) / 4u + 1; index += index__; value__ >>= 4u * index__;{code}\
if (Group::nItems >= 4 && value__) {size_t index__ = __builtin_ctzll(value__) / 4u + 1; index += index__; value__ >>= 4u * index__;{code}\
if (Group::nItems >= 5 && value__) {size_t index__ = __builtin_ctzll(value__) / 4u + 1; index += index__; value__ >>= 4u * index__;{code}\
if (Group::nItems >= 6 && value__) {size_t index__ = __builtin_ctzll(value__) / 4u + 1; index += index__; value__ >>= 4u * index__;{code}\
if (Group::nItems >= 7 && value__) {size_t index__ = __builtin_ctzll(value__) / 4u + 1; index += index__; value__ >>= 4u * index__;{code}\
}}}}}}}}


#define forEachMatch(distances, key, code) {\
  auto k0 = group->arr[0].key == key; \
  auto k1 = Group::nItems >= 2 && group->arr[1].key == key; \
  auto k2 = Group::nItems >= 3 && group->arr[2].key == key; \
  auto k3 = Group::nItems >= 4 && group->arr[3].key == key; \
  auto k4 = Group::nItems >= 5 && group->arr[4].key == key; \
  auto k5 = Group::nItems >= 6 && group->arr[5].key == key; \
  auto k6 = Group::nItems >= 7 && group->arr[6].key == key; \
  auto s0 = bool(distances & 0xf) && k0; \
  auto s1 = Group::nItems >= 2 && (bool(distances & 0xf0) && k1); \
  auto s2 = Group::nItems >= 3 && (bool(distances & 0xf00) && k2); \
  auto s3 = Group::nItems >= 4 && (bool(distances & 0xf000) && k3); \
  auto s4 = Group::nItems >= 5 && (bool(distances & 0xf0000) && k4); \
  auto s5 = Group::nItems >= 6 && (bool(distances & 0xf00000) && k5); \
  auto s6 = Group::nItems >= 7 && (bool(distances & 0xf000000) && k6); \
  if (k0 | k1 | k2 | k3 | k4 | k5 | k6) {\
    if (s0) {uint8_t index = 0; code}\
    if (s1) {uint8_t index = 1; code}\
    if (s2) {uint8_t index = 2; code}\
    if (s3) {uint8_t index = 3; code}\
    if (s4) {uint8_t index = 4; code}\
    if (s5) {uint8_t index = 5; code}\
    if (s6) {uint8_t index = 6; code}\
  }\
}

  template<typename KeyT, size_t findFirstFreeSlot = false>
  [[gnu::always_inline]] [[gnu::hot]]
  std::conditional_t<findFirstFreeSlot, std::pair<iterator, iterator>, iterator> find(KeyT&& key) const noexcept {
    if (unlikely(!groups)) {
      if constexpr (findFirstFreeSlot) {
        return std::make_pair(end(), end());
      } else {
        return end();
      }
    }
    size_t bs = ksize;
    size_t mask = bs - 1;
    size_t gi = Hash()(key) & mask;
    size_t originalGi = gi;

    //printf("find key %d at gi %d\n", key, gi);

    Group* groups = this->groups;
    
    Group* group = indexGroup(groups, gi);
    uint8_t maxDistance = group->ext >> 28;
    uint32_t distances = group->ext & 0xfffffff;

    //printf("maxDistance is %d\n", maxDistance);

    size_t ei = (gi + maxDistance) & mask;
    iterator free(nullptr, none, none);
    while (true) {
      uint8_t set = 0;
      if constexpr (true || findFirstFreeSlot) {
        forEachSet(distances,
          if constexpr (findFirstFreeSlot) {
            set |= (uint8_t)1 << index;
          }
          if (group->arr[index].key == key) {
            if constexpr (findFirstFreeSlot) {
              return std::make_pair(iterator(this, gi, index), free);
            } else {
              return iterator(this, gi, index);
            }
          }
        );
      } else {
        forEachMatch(distances, key,
          return iterator(this, gi, index);
        );
      }

      if constexpr (findFirstFreeSlot) {
        set ^= ((uint8_t)1 << Group::nItems) - 1;
        if (set) {
          size_t index = __builtin_ctzll(set);
          free = iterator(this, gi, index);
        }
      }

      //printf("not found in %d\n", gi);
      if (gi == ei) {
        break;
      }
      gi = (gi + 1) & mask;
      group = indexGroup(groups, gi);
      distances = group->ext & 0xfffffff;
    }
    if constexpr (findFirstFreeSlot) {
      iterator i(this, originalGi, none);
      if (!free.map) {
        //printf("find further free!\n");
        gi = (gi + 1) & mask;
        group = indexGroup(groups, gi);
        distances = group->ext & 0xfffffff;
        while (true) {
          uint8_t set = 0;
          forEachSet(distances,
            set |= (uint8_t)1 << index;
          );

          set ^= ((uint8_t)1 << Group::nItems) - 1;
          if (set) {
            size_t index = __builtin_ctzll(set);
            free = iterator(this, gi, index);
            return std::make_pair(i, free);
          }
          //printf("try gi %d   (%d %d %d)\n", gi, ksize, msize, std::distance(begin(), end()));
          // size_t prevIndex = -1;
          // if constexpr (findFirstFreeSlot) {
          //   if (!free.map) {
          //     if (distances == 0) {
          //       free = iterator(this, gi, 0);
          //       return std::make_pair(i, free);
          //     }
          //   }
          // }
          // while (distances) {
          //   uint8_t index = __builtin_ctzll(distances);
          //   index /= 4u;
          //   distances &= ~(0xfu << (4u * index));
          //   //printf("found something at %d\n", index);
          //   if (index != prevIndex + 1) {
          //     free = iterator(this, gi, prevIndex + 1);
          //     return std::make_pair(i, free);
          //   }
          //   prevIndex = index;
          // }
          // if (prevIndex != Group::nItems - 1) {
          //   free = iterator(this, gi, prevIndex + 1);
          //   return std::make_pair(i, free);
          // }
          gi = (gi + 1) & mask;
          group = indexGroup(groups, gi);
          distances = group->ext & 0xfffffff;
        }
      }
      return std::make_pair(i, free);
    } else {
      return end();
    }
  }

  template<typename KeyT, size_t findFirstFreeSlot = false>
  std::conditional_t<findFirstFreeSlot, std::pair<iterator, iterator>, iterator> findOld(KeyT&& key) const noexcept {
    if (unlikely(!groups)) {
      if constexpr (findFirstFreeSlot) {
        return std::make_pair(end(), end());
      } else {
        return end();
      }
    }
    size_t bs = ksize;
    size_t mask = bs - 1;
    size_t gi = Hash()(key) & mask;
    size_t originalGi = gi;

    //printf("find key %d at gi %d\n", key, gi);

    Group* groups = this->groups;
    
    Group* group = indexGroup(groups, gi);
    uint8_t maxDistance = group->ext >> 28;
    uint32_t distances = group->ext & 0xfffffff;

    //printf("maxDistance is %d\n", maxDistance);

    size_t ei = (gi + maxDistance) & mask;
    iterator free(nullptr, none, none);
    while (true) {
      size_t prevIndex = -1;
      if constexpr (findFirstFreeSlot) {
        if (!free.map) {
          if (distances == 0) {
            //printf("find free at 0 cuz 0 distanes!\n");
            free = iterator(this, gi, 0);
          }
        }
      }
      while (distances) {
        uint8_t index = __builtin_ctzll(distances);
        index /= 4u;
        distances &= ~(0xfu << (4u * index));
        if constexpr (findFirstFreeSlot) {
          if (!free.map) {
            if (index != prevIndex + 1) {
              //printf("find free at %d cuz distance %d\n", prevIndex + 1, index - prevIndex);
              free = iterator(this, gi, prevIndex + 1);
            }
            prevIndex = index;
          }
        }
        auto& v = group->arr[index];
        if (v.key == key) {
          if constexpr (findFirstFreeSlot) {
            return std::make_pair(iterator(this, gi, index), free);
          } else {
            return iterator(this, gi, index);
          }
        }
      }
      if constexpr (findFirstFreeSlot) {
        if (!free.map) {
          if (prevIndex != Group::nItems - 1) {
            free = iterator(this, gi, prevIndex + 1);
          }
        }
      }
      //printf("not found in %d\n", gi);
      if (gi == ei) {
        break;
      }
      gi = (gi + 1) & mask;
      group = indexGroup(groups, gi);
      distances = group->ext & 0xfffffff;
    }
    if constexpr (findFirstFreeSlot) {
      iterator i(this, originalGi, none);
      if (!free.map) {
        //printf("find further free!\n");
        gi = (gi + 1) & mask;
        group = indexGroup(groups, gi);
        distances = group->ext & 0xfffffff;
        while (true) {
          //printf("try gi %d   (%d %d %d)\n", gi, ksize, msize, std::distance(begin(), end()));
          size_t prevIndex = -1;
          if constexpr (findFirstFreeSlot) {
            if (!free.map) {
              if (distances == 0) {
                free = iterator(this, gi, 0);
                return std::make_pair(i, free);
              }
            }
          }
          while (distances) {
            uint8_t index = __builtin_ctzll(distances);
            index /= 4u;
            distances &= ~(0xfu << (4u * index));
            //printf("found something at %d\n", index);
            if (index != prevIndex + 1) {
              free = iterator(this, gi, prevIndex + 1);
              return std::make_pair(i, free);
            }
            prevIndex = index;
          }
          if (prevIndex != Group::nItems - 1) {
            free = iterator(this, gi, prevIndex + 1);
            return std::make_pair(i, free);
          }
          gi = (gi + 1) & mask;
          group = indexGroup(groups, gi);
          distances = group->ext & 0xfffffff;
        }
      }
      return std::make_pair(i, free);
    } else {
      return end();
    }
  }

  template<typename KeyT>
  std::pair<iterator, iterator> findFree(KeyT&& key) const noexcept {
    return find<KeyT, true>(std::forward<KeyT>(key));
  }

  void reserve(size_t n) {
    if (n >= std::numeric_limits<size_t>::max() / 2) {
      throw std::range_error("reserve beyond max size");
    }
    size_t bs = ksize;
    if (bs == 0) {
      bs = 1;
    }
    while (bs < n) {
      bs *= 2;
    }
    rehash(bs);
  }

  template<typename KeyT, typename... Args>
  std::pair<iterator, bool> try_emplace(KeyT&& key, Args&&... args) {
    if (unlikely(!groups)) {
      reserve(1);
    }
    auto [i, free] = findFree(key);
    assert(!isNone(i.gi));
    if (!isNone(i.vi)) {
      return std::make_pair(i, false);
    }
    if (unlikely(ksize * Group::nItems <= msize + 1)) {
      size_t omsize = msize;
      reserve(ksize + 1);
      std::tie(i, free) = findFree(key);
      assert(isNone(i.vi));
      assert(!isNone(free.vi));
      assert(msize == omsize);
    }
    assert(free.gi < ksize);
    assert(free.vi < Group::nItems);
    size_t gi = i.gi;
    Group* group = indexGroup(groups, free.gi);
    assert((group->ext & (0xfu << (4u * free.vi))) == 0);
    size_t distance = (free.gi - gi) & (ksize - 1);
    //printf("insert new key %d at %d %d, distance %d\n", key, free.gi, free.vi, distance);
    while (distance > 7) {
      // for (size_t ii = 0; ii != ksize; ++ii) {
      //   Group* group = indexGroup(groups, ii);
      //   printf("group %d  distances %#x  (max distance %d)\n", ii, group->ext & 0xfffffff, group->ext >> 28);
      // }
      //printf("rehashing due to distance > 15 :(\n");
      //std::abort();
      size_t omsize = msize;
      reserve(ksize + 1);
      std::tie(i, free) = findFree(key);
      assert(isNone(i.vi));
      assert(!isNone(free.vi));
      assert(msize == omsize);

      gi = i.gi;
      group = indexGroup(groups, free.gi);
      assert((group->ext & (0xfu << (4u * free.vi))) == 0);
      distance = (free.gi - gi) & (ksize - 1);
    }
    // for (size_t ii = i.gi;; ii = (ii + 1) & (ksize - 1)) {
    //   Group* group = indexGroup(groups, ii);
    //   printf("group %d  distances %#x  (max distance %d)\n", ii, group->ext & 0xfffffff, group->ext >> 28);
    //   if (ii == free.gi) {
    //     break;
    //   }
    // }
    // if (distance > 14) {
    //   printf("distance is %d :(\n", distance);
    //   std::abort();
    // }
    ++msize;
    assert(((group->ext >> (4u * free.vi)) & 0xf) == 0);
    group->ext |= (uint8_t)(distance + 1) << (4u * free.vi);
    //printf("group->ext is now %#x\n", group->ext);
    if (distance != 0) {
      Group* ogroup = indexGroup(groups, i.gi);
      uint8_t maxDistance = ogroup->ext >> 28u;
      if (distance > maxDistance) {
        ogroup->ext = (ogroup->ext & 0xfffffffu) | ((uint32_t)distance << 28u);
      }
    }
    assert((group->ext & (0xfu << (4u * free.vi))) != 0);
    auto& v = group->arr[free.vi];
    new (&v.key) Key(std::forward<KeyT>(key));
    new (&v.value) Value(std::forward<Args>(args)...);
    // for (auto i = begin(); i != end(); ++i) {
    //   printf("  %d %d -> %d\n", i.gi, i.vi, i->first);
    // }
    return std::make_pair(free, true);
  }

  template<typename... Args>
  auto emplace(Args&&... args) {
    return try_emplace(std::forward<Args>(args)...);
  }

  size_t bucket_count() const noexcept {
    return ksize;
  }
  size_t size() const noexcept {
    return msize;
  }

};

}

#undef unrollSetBits
