#pragma once

#include <cstddef>
#include <iterator>
#include <cstring>
#include <cassert>

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

template<typename Key, typename Value, typename Hash = std::hash<Key>>
struct HashMap {
private:
  size_t ksize = 0;
  size_t msize = 0;
  bool* hasKey = nullptr;
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
      //   if (!map->hasKey[ki]) {
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
        } while (!map->hasKey[ki]);
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
  }
  iterator begin() noexcept {
    if (!hasKey) {
      return end();
    }
    size_t bs = ksize;
    for (size_t i = 0; i != bs; ++i) {
      if (hasKey[i]) {
        return iterator(this, i, -1, &values[i]);
      }
    }
    for (size_t i = 0; i != bs; ++i) {
      if (indices2[i] != -1) {
        return iterator(this, i, i, &values2[i]);
      }
    }
    return end();
  }
  iterator end() noexcept {
    return iterator(this, 0, 0, nullptr);
  }

  template<typename T>
  T* allocate(size_t n) {
    return (T*)std::aligned_alloc(std::max(alignof(T), (size_t)32), sizeof(T) * n);
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
        sizes2[ki] = s;
        return i;
      }
      ++i;
      hasKey[ki] = false;
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
      sizes2[ki] = s;
      return i;
    }
  }

  template<typename KeyT, typename ValueT>
  iterator insert(KeyT&& key, ValueT&& value) {
    return try_emplace(std::forward<Key>(key), std::forward<Value>(value)).first;
  }

  void rehash1(size_t newBs) noexcept {
    if (newBs & (newBs - 1)) {
      printf("bucket count is not a multiple of 2!\n");
      std::abort();
    }
    //printf("rehash %d %d\n", newBs, size());
    bool* oldHasKey = hasKey;
    Key* oldKeys = keys;
    Value* oldValues = values;

    hasKey = allocate<bool>(newBs);
    keys = allocate<Key>(newBs);
    values = allocate<Value>(newBs);

    std::memset(hasKey, 0, newBs);

    if (oldHasKey) {
      size_t bs = ksize;
      ksize = newBs;
      for (size_t i = 0; i != bs; ++i) {
        if (oldHasKey[i]) {
          insert(std::move(oldKeys[i]), std::move(oldValues[i]));
          --msize;
          oldKeys[i].~Key();
          oldValues[i].~Value();
        }
      }
      ksize = bs;
    }

    deallocate(oldHasKey);
    deallocate(oldKeys);
    deallocate(oldValues);
  }

  void rehash2(size_t newBs) noexcept {
    if (newBs & (newBs - 1)) {
      printf("bucket count is not a multiple of 2!\n");
      std::abort();
    }
    Key* oldKeys2 = keys2;
    Value* oldValues2 = values2;
    size_t* oldSizes2 = sizes2;
    size_t* oldIndices2 = indices2;

    keys2 = allocate<Key>(newBs);
    values2 = allocate<Value>(newBs);
    sizes2 = allocate<size_t>(newBs);
    indices2 = allocate<size_t>(newBs);

    std::memset(sizes2, 0, sizeof(size_t) * newBs);
    std::memset(indices2, -1, sizeof(size_t) * newBs);

    if (oldKeys2) {
      size_t bs = ksize;
      ksize = newBs;
      for (size_t i = 0; i != bs; ++i) {
        if (oldIndices2[i] != -1) {
          insert(std::move(oldKeys2[i]), std::move(oldValues2[i]));
          --msize;
          oldKeys2[i].~Key();
          oldValues2[i].~Value();
        }
      }
      ksize = bs;
    }

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
      if (likely(hasKey[ki])) {
        if (likely(keys[ki] == key)) {
          return iterator(this, ki, -1, &values[ki]);
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
    size_t bs = ksize;
    if (bs == 0) {
      bs = 1;
    }
    while (bs < n) {
      bs *= 2;
    }
    rehash1(bs);
    rehash2(bs);
    ksize = bs;
  }

  template<typename KeyT, typename... Args>
  std::pair<iterator, bool> try_emplace(KeyT&& key, Args&&... args) {
    if (!hasKey) {
      reserve(16);
    }
    auto i = find(key);
    if (i.v) {
      return std::make_pair(i, false);
    }
    ++msize;
    if (ksize < msize) {
      reserve(msize);
      i = find(key);
      assert(i.v == nullptr);
    }
    if (i.vi == -1) {
      auto* keys = this->keys;
      auto* values = this->values;
      hasKey[i.ki] = true;
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

}
