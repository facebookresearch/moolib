#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iterator>

#if 0
#define likely(x) x
#define unlikely(x) x
#define aligned(x, alignment) x
#else
#define likely(x) __builtin_expect(bool(x), 1)
#define unlikely(x) __builtin_expect(bool(x), 0)

#define aligned(x, alignment) (std::remove_reference_t<decltype((x))>)__builtin_assume_aligned((x), alignment)
#endif

namespace moolib {

bool getBit(uint64_t* ptr, size_t index) {
  return ptr[index / 64] & (1ull << (index % 64));
}
void setBit(uint64_t* ptr, size_t index, bool value) {
  if (value) {
    ptr[index / 64] |= (1ull << (index % 64));
  } else {
    ptr[index / 64] &= ~(1ull << (index % 64));
  }
}
void setBit(uint64_t* ptr, size_t index) {
  ptr[index / 64] |= (1ull << (index % 64));
}
void unsetBit(uint64_t* ptr, size_t index) {
  ptr[index / 64] &= ~(1ull << (index % 64));
}

#define unrollSetBits(inputvalue, code)                                                                                \
  if (inputvalue) {                                                                                                    \
    uint64_t value__ = inputvalue;                                                                                     \
    size_t index = __builtin_ctzll(value__);                                                                           \
    value__ >>= index + 1;                                                                                             \
    { code }                                                                                                           \
    if (value__) {                                                                                                     \
      size_t index__ = __builtin_ctzll(value__) + 1;                                                                   \
      index += index__;                                                                                                \
      value__ >>= index__;                                                                                             \
      { code }                                                                                                         \
      if (value__) {                                                                                                   \
        size_t index__ = __builtin_ctzll(value__) + 1;                                                                 \
        index += index__;                                                                                              \
        value__ >>= index__;                                                                                           \
        { code }                                                                                                       \
        if (value__) {                                                                                                 \
          size_t index__ = __builtin_ctzll(value__) + 1;                                                               \
          index += index__;                                                                                            \
          value__ >>= index__;                                                                                         \
          { code }                                                                                                     \
          if (value__) {                                                                                               \
            size_t index__ = __builtin_ctzll(value__) + 1;                                                             \
            index += index__;                                                                                          \
            value__ >>= index__;                                                                                       \
            { code }                                                                                                   \
            if (value__) {                                                                                             \
              size_t index__ = __builtin_ctzll(value__) + 1;                                                           \
              index += index__;                                                                                        \
              value__ >>= index__;                                                                                     \
              { code }                                                                                                 \
              if (value__) {                                                                                           \
                size_t index__ = __builtin_ctzll(value__) + 1;                                                         \
                index += index__;                                                                                      \
                value__ >>= index__;                                                                                   \
                { code }                                                                                               \
                if (value__) {                                                                                         \
                  size_t index__ = __builtin_ctzll(value__) + 1;                                                       \
                  index += index__;                                                                                    \
                  value__ >>= index__;                                                                                 \
                  { code }                                                                                             \
                  for (int i__ = 14; i__; --i__) {                                                                     \
                    if (!value__) break;                                                                               \
                    {                                                                                                  \
                      size_t index__ = __builtin_ctzll(value__) + 1;                                                   \
                      index += index__;                                                                                \
                      value__ >>= index__;                                                                             \
                      { code }                                                                                         \
                    }                                                                                                  \
                    if (!value__) break;                                                                               \
                    {                                                                                                  \
                      size_t index__ = __builtin_ctzll(value__) + 1;                                                   \
                      index += index__;                                                                                \
                      value__ >>= index__;                                                                             \
                      { code }                                                                                         \
                    }                                                                                                  \
                    if (!value__) break;                                                                               \
                    {                                                                                                  \
                      size_t index__ = __builtin_ctzll(value__) + 1;                                                   \
                      index += index__;                                                                                \
                      value__ >>= index__;                                                                             \
                      { code }                                                                                         \
                    }                                                                                                  \
                    if (!value__) break;                                                                               \
                    {                                                                                                  \
                      size_t index__ = __builtin_ctzll(value__) + 1;                                                   \
                      index += index__;                                                                                \
                      value__ >>= index__;                                                                             \
                      { code }                                                                                         \
                    }                                                                                                  \
                  }                                                                                                    \
                }                                                                                                      \
              }                                                                                                        \
            }                                                                                                          \
          }                                                                                                            \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

template<typename Key, typename Value, typename Hash = std::hash<Key>>
struct HashMap {
private:
  size_t ksize = 0;
  size_t msize = 0;
  uint64_t* hasKey = nullptr;
  uint64_t* hasKey2 = nullptr;
  Key* keys = nullptr;
  Value* values = nullptr;
  Key* keys2 = nullptr;
  size_t* indices2 = nullptr;
  Value* values2 = nullptr;
  size_t* sizes2 = nullptr;

public:
  struct iterator {
  private:
  public:
    friend HashMap;
    HashMap* map;
    size_t ki;
    size_t vi;
    Value* v;

  public:
    iterator() = default;
    iterator(HashMap* map, size_t ki, size_t vi, Value* v) : map(map), ki(ki), vi(vi), v(v) {}

    using T = Value;

    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::bidirectional_iterator_tag;

    T& operator*() const noexcept {
      // if (!v) {
      //   printf("deferencing end tensor\n");
      //   std::abort();
      // }
      // size_t bs = map->ksize;
      // if (bs == 0) {
      //   printf("map is empty!\n");
      //   std::abort();
      // }
      // if (vi == -1) {
      //   if (ki >= bs) {
      //     printf("out of bounds ki! (%d/%d)\n", ki, bs);
      //     std::abort();
      //   }
      //   if (!getBit(map->hasKey, ki)) {
      //     printf("map does not have key! %d\n", ki);
      //     std::abort();
      //   }
      //   if (v != &map->values[ki]) {
      //     printf("v ki mismatch\n");
      //     std::abort();
      //   }
      // } else {
      //   if (vi >= map->ksize) {
      //     printf("out of bounds vi! (%d/%d)\n", vi, map->ksize);
      //     std::abort();
      //   }
      //   if (v != &map->values2[vi]) {
      //     printf("v vi mismatch\n");
      //     std::abort();
      //   }
      // }
      return *v;
    }
    T* operator->() const noexcept {
      return &**this;
    }

    iterator& operator++() noexcept {
      **this;
      if (vi == -1) {
        do {
          if (ki == map->ksize - 1) {
            ki = -1;
            vi = 0;
            v = &map->values2[vi];
            if (map->indices2[vi] != -1) {
              return *this;
            }
            return ++*this;
          } else {
            ++ki;
            ++v;
          }
        } while (!getBit(map->hasKey, ki));
      } else {
        do {
          if (vi == map->ksize - 1) {
            v = nullptr;
            return *this;
          }
          ++vi;
          ++v;
        } while (map->indices2[vi] == -1);
      }
      return *this;
    }
    iterator operator++(int) noexcept {
      iterator r = *this;
      ++r;
      return r;
    }
    bool operator==(iterator n) const noexcept {
      return v == n.v;
    }
    bool operator!=(iterator n) const noexcept {
      return v != n.v;
    }
  };

  HashMap() = default;
  HashMap(const HashMap&) = delete;
  HashMap& operator=(const HashMap&) = delete;
  ~HashMap() {
    clear();
    deallocate(hasKey);
    deallocate(hasKey2);
    deallocate(keys);
    deallocate(values);
    deallocate(keys2);
    deallocate(values2);
    deallocate(sizes2);
    deallocate(indices2);
  }

  void clear() noexcept {
    auto i = begin();
    auto e = end();
    while (i != e) {
      i = erase(i);
    }
    assert(msize == 0);
    assert(begin() == end());
  }
  iterator begin() noexcept {
    if (!hasKey) {
      return end();
    }
    size_t bs = ksize;
    for (size_t i = 0; i < bs; i += 64) {
      unrollSetBits(hasKey[i / 64], return iterator(this, i + index, -1, &values[i + index]););
    }
    for (size_t i = 0; i < bs; i += 64) {
      unrollSetBits(hasKey2[i / 64], i += index; return iterator(this, i, i, &values2[i]););
    }
    return end();
  }
  iterator end() noexcept {
    return iterator(this, 0, 0, nullptr);
  }

  template<typename T>
  T* allocate(size_t n) {
    T* retval = (T*)std::aligned_alloc(std::max(alignof(T), (size_t)32), std::max(sizeof(T) * n, (size_t)32));
    std::memset(retval, 0, std::max(sizeof(T) * n, (size_t)32));
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

  iterator erase(iterator i) noexcept {
    auto mask = ksize - 1;
    size_t ki = i.ki;
    size_t vi = i.vi;
    assert(msize > 0);
    --msize;
    auto* keys2 = this->keys2;
    auto* values2 = this->values2;
    auto* sizes2 = this->sizes2;
    if (vi == -1) {
      size_t s = sizes2[ki];
      if (s) {
        --s;
        size_t index = (ki + s) & mask;
        Key* k = &keys2[index];
        Value* v = &values2[index];
        indices2[index] = -1;
        keys[ki] = std::move(*k);
        values[ki] = std::move(*v);
        k->~Key();
        v->~Value();
        while (s && indices2[(ki + s - 1) & mask] != ki) {
          --s;
        }
        if (s == 0) {
          unsetBit(hasKey2, ki);
        }
        sizes2[ki] = s;
        return i;
      }
      ++i;
      unsetBit(hasKey, ki);
      // printf("unset %d, hasKey[%d] is now %#x\n", ki, ki/64, hasKey[ki / 64]);
      keys[ki].~Key();
      values[ki].~Value();
      if (i != end()) {
        *i;
      }
      return i;
    } else {
      if (ki == -1) {
        ki = indices2[vi];
      }
      size_t s = sizes2[ki] - 1;
      size_t lastIndex = (ki + s) & mask;
      if (lastIndex == vi) {
        ++i;
      } else {
        keys2[vi] = std::move(keys2[lastIndex]);
        values2[vi] = std::move(values2[lastIndex]);
      }
      indices2[lastIndex] = -1;
      keys2[lastIndex].~Key();
      values2[lastIndex].~Value();
      while (s && indices2[(ki + s - 1) & mask] != ki) {
        --s;
      }
      if (s == 0) {
        unsetBit(hasKey2, ki);
      }
      sizes2[ki] = s;
      return i;
    }
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
    // printf("rehash %d %d\n", newBs, size());
    uint64_t* oldHasKey = hasKey;
    uint64_t* oldHasKey2 = hasKey2;
    Key* oldKeys = keys;
    Value* oldValues = values;
    Key* oldKeys2 = keys2;
    Value* oldValues2 = values2;
    size_t* oldSizes2 = sizes2;
    size_t* oldIndices2 = indices2;

    hasKey = allocate<uint64_t>((newBs + 63) / 64);
    hasKey2 = allocate<uint64_t>((newBs + 63) / 64);
    keys = allocate<Key>(newBs);
    values = allocate<Value>(newBs);
    keys2 = allocate<Key>(newBs);
    values2 = allocate<Value>(newBs);
    sizes2 = allocate<size_t>(newBs);
    indices2 = allocate<size_t>(newBs);

    std::memset(hasKey, 0, sizeof(uint64_t) * ((newBs + 63) / 64));
    std::memset(hasKey2, 0, sizeof(uint64_t) * ((newBs + 63) / 64));
    std::memset(sizes2, 0, sizeof(size_t) * newBs);
    std::memset(indices2, -1, sizeof(size_t) * newBs);

    if (oldHasKey) {
      size_t bs = ksize;
      ksize = newBs;
      for (size_t i = 0; i < bs; i += 64) {
        unrollSetBits(oldHasKey[i / 64], size_t ki = i + index; --msize;
                      try_emplace(std::move(oldKeys[ki]), std::move(oldValues[ki])); oldKeys[ki].~Key();
                      oldValues[ki].~Value(););
      }
      for (size_t vi = 0; vi != bs; ++vi) {
        if (oldIndices2[vi] != -1) {
          --msize;
          try_emplace(std::move(oldKeys2[vi]), std::move(oldValues2[vi]));
          oldKeys2[vi].~Key();
          oldValues2[vi].~Value();
        }
      }
      ksize = bs;
    }

    deallocate(oldHasKey);
    deallocate(oldHasKey2);
    deallocate(oldKeys);
    deallocate(oldValues);
    deallocate(oldKeys2);
    deallocate(oldValues2);
    deallocate(oldSizes2);
    deallocate(oldIndices2);
  }

  template<typename KeyT>
  iterator find(KeyT&& key) noexcept {
    size_t bs = ksize;
    size_t mask = bs - 1;
    size_t ki = Hash()(key) & mask;
    auto* hasKey = aligned(this->hasKey, 32);
    auto* keys = aligned(this->keys, 32);
    auto* sizes2 = aligned(this->sizes2, 32);
    auto* indices2 = aligned(this->indices2, 32);
    auto* keys2 = aligned(this->keys2, 32);
    auto* values = aligned(this->values, 32);
    auto* values2 = aligned(this->values2, 32);
    if (likely(hasKey)) {
      if (unlikely(getBit(hasKey, ki))) {
        if (likely(keys[ki] == key)) {
          return iterator(this, ki, -1, &values[ki]);
        } else if (likely(!getBit(hasKey2, ki))) {
          return iterator(this, ki, ki, nullptr);
        }
      } else {
        return iterator(this, ki, -1, nullptr);
      }
    } else {
      return iterator(this, ki, -1, nullptr);
    }
    size_t s = sizes2[ki];
    size_t index = ki + s;
    while (index != ki) {
      --index;
      size_t i = index & mask;
      if (likely(indices2[i] == ki && keys2[i] == key)) {
        return iterator(this, ki, i, &values2[i]);
      }
    }
    return iterator(this, ki, (ki + s) & mask, nullptr);
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
    ksize = bs;
  }

  template<typename KeyT, typename... Args>
  std::pair<iterator, bool> try_emplace(KeyT&& key, Args&&... args) {
    if (unlikely(!hasKey)) {
      reserve(16);
    }
    auto i = find(key);
    if (unlikely(i.v)) {
      return std::make_pair(i, false);
    }
    ++msize;
    if (unlikely(ksize < msize)) {
      size_t omsize = msize;
      reserve(msize);
      i = find(key);
      assert(i.v == nullptr);
      assert(msize == omsize);
    }
    if (likely(i.vi == -1)) {
      auto* keys = this->keys;
      auto* values = this->values;
      setBit(hasKey, i.ki);
      new (&keys[i.ki]) Key(std::forward<KeyT>(key));
      new (&values[i.ki]) Value(std::forward<Args>(args)...);
      return std::make_pair(iterator(this, i.ki, -1, &values[i.ki]), true);
    } else {
      auto* indices2 = this->indices2;
      auto* sizes2 = this->sizes2;
      auto* keys2 = this->keys2;
      auto* values2 = this->values2;
      size_t mask = ksize - 1;
      size_t s = sizes2[i.ki] + 1;
      size_t index = i.vi;
      if (s == 1) {
        setBit(hasKey2, i.ki);
      }
      while (indices2[index] != -1) {
        index = (index + 1) & mask;
        ++s;
      }
      sizes2[i.ki] = s;
      indices2[index] = i.ki;
      new (&keys2[index]) Key(std::forward<KeyT>(key));
      new (&values2[index]) Value(std::forward<Args>(args)...);
      return std::make_pair(iterator(this, i.ki, index, &values2[index]), true);
    }
  }

  size_t bucket_count() const noexcept {
    return ksize;
  }
  size_t size() const noexcept {
    return msize;
  }
};

} // namespace moolib

#undef unrollSetBits
