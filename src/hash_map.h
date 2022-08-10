#pragma once

#include <cstddef>
#include <iterator>
#include <cstring>

//#include "fmt/printf.h"

namespace moolib {

#if 0
template<typename Key, typename Value, typename Hash = std::hash<Key>>
struct HashMap {
private:
  std::unordered_map<Key, Value> map;
public:
  struct iterator {
  private:
  public:
    friend HashMap;
    typename std::unordered_map<Key, Value>::iterator i;
  public:
    iterator() = default;
    iterator(typename std::unordered_map<Key, Value>::iterator i) : i(i) {}

    using T = Value;

    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::bidirectional_iterator_tag;

    T& operator*() const noexcept {
      return i->second;
    }
    T* operator->() const noexcept {
      return &**this;
    }
    iterator& operator++() noexcept {
      ++i;
      return *this;
    }
    iterator operator++(int) noexcept {
      iterator r = *this;
      ++r;
      return r;
    }
    iterator& operator--() noexcept {
      --i;
      return *this;
    }
    iterator operator--(int) noexcept {
      iterator r = *this;
      --r;
      return r;
    }
    bool operator==(iterator n) const noexcept {
      return i == n.i;
    }
    bool operator!=(iterator n) const noexcept {
      return i != n.i;
    }
  };


  void clear() noexcept {
    map.clear();
  }
  iterator begin() noexcept {
    return map.begin();
  }
  iterator end() noexcept {
    return map.end();
  }

  iterator erase(iterator i) noexcept {
    return map.erase(i.i);
  }

  template<typename KeyT, typename ValueT>
  iterator insert(KeyT&& key, ValueT&& value) {
    return try_emplace(std::forward<Key>(key), std::forward<Value>(value)).first;
  }

  template<typename KeyT>
  iterator find(KeyT&& key) noexcept {
    return map.find(std::forward<KeyT>(key));
  }

  template<typename KeyT, typename... Args>
  std::pair<iterator, bool> try_emplace(KeyT&& key, Args&&... args) {
    return map.try_emplace(std::forward<KeyT>(key), std::forward<Args>(args)...);
  }

  size_t bucket_count() const noexcept {
    return map.bucket_count();
  }
  size_t size() const noexcept {
    return map.size();
  }

};

#else
template<typename Key, typename Value, typename Hash = std::hash<Key>>
struct HashMap {
private:
  size_t ksize = 0;
  size_t msize = 0;
  bool* hasKey = nullptr;
  Key* keys = nullptr;
  Value* values = nullptr;
  Key** keys2 = nullptr;
  Value** values2 = nullptr;
  size_t* sizes2 = nullptr;
  size_t* allocated2 = nullptr;
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
      //   fmt::printf("deferencing end tensor\n");
      //   std::abort();
      // }
      // size_t bs = map->ksize;
      // if (bs == 0) {
      //   fmt::printf("map is empty!\n");
      //   std::abort();
      // }
      // if (ki >= bs) {
      //   fmt::printf("out of bounds ki! (%d/%d)\n", ki, bs);
      //   std::abort();
      // }
      // if (vi == -1) {
      //   if (!map->hasKey[ki]) {
      //     fmt::printf("map does not have key! %d\n", ki);
      //     std::abort();
      //   }
      //   if (v != &map->values[ki]) {
      //     fmt::printf("v ki mismatch\n");
      //     std::abort();
      //   }
      // } else {
      //   if (vi >= map->sizes2[ki]) {
      //     fmt::printf("out of bounds vi! (%d/%d)\n", vi, map->sizes2[ki]);
      //     std::abort();
      //   }
      //   if (v != &map->values2[ki][vi]) {
      //     fmt::printf("v vi mismatch\n");
      //     std::abort();
      //   }
      // }
      return *v;
    }
    T* operator->() const noexcept {
      return &**this;
    }
    void prevValid() {
      do {
        if (ki == 0) {
          ki = map->ksize - 1;
          vi = -1;
          v = map->values[ki];
          return;
        }
        --ki;
      } while (map->sizes2[ki] == 0);
      vi = map->sizes2[ki] - 1;
      v = &map->values2[ki][vi];
    }
    void nextValid() {
      size_t s = map->ksize;
      vi = 0;
      do {
        ++ki;
        if (ki == s) {
          v = nullptr;
          return;
        }
      } while (map->sizes2[ki] == 0);
      v = &map->values2[ki][0];
      **this;
    }
    iterator& operator++() noexcept {
      if (vi == -1) {
        do {
          if (ki == map->ksize - 1) {
            ki = -1;
            vi = 0;
            nextValid();
            break;
          } else {
            ++ki;
            ++v;
          }
        } while (!map->hasKey[ki]);
      } else {
        if (vi == map->sizes2[ki] - 1) {
          nextValid();
          if (v) {
            **this;
          }
        } else {
          ++vi;
          ++v;
        }
      }
      return *this;
    }
    iterator operator++(int) noexcept {
      iterator r = *this;
      ++r;
      return r;
    }
    iterator& operator--() noexcept {
      if (vi == -1) {
        do {
          if (ki == 0) {
            v = nullptr;
            break;
          } else {
            --ki;
            --v;
          }
        } while (!map->hasKey[ki]);
      } else {
        if (vi == 0) {
          prevValid();
        } else {
          --vi;
          --v;
        }
      }
      return *this;
    }
    iterator operator--(int) noexcept {
      iterator r = *this;
      --r;
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
    if (keys2) {
      size_t bs = ksize;
      for (size_t ki = 0; ki != bs; ++ki) {
        if (keys2[ki]) {
          deallocate(keys2[ki]);
          deallocate(values2[ki]);
        }
      }
    }
    deallocate(hasKey);
    deallocate(keys);
    deallocate(values);
    deallocate(keys2);
    deallocate(values2);
    deallocate(sizes2);
    deallocate(allocated2);
  }

  void clear() noexcept {
    auto i = begin();
    auto e = end();
    while (i != e) {
      i = erase(i);
    }
  }
  iterator begin() noexcept {
    size_t bs = ksize;
    if (bs == 0) {
      return end();
    }
    for (size_t i = 0; i != bs; ++i) {
      if (hasKey[i]) {
        return iterator(this, i, -1, &values[i]);
      }
    }
    auto r = iterator(this, -1, 0, nullptr);
    r.nextValid();
    *r;
    return r;
  }
  iterator end() noexcept {
    return iterator(this, ksize, 0, nullptr);
  }

  template<typename T>
  T* allocate(size_t n) {
    return (T*)std::aligned_alloc(alignof(T), sizeof(T) * n);
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
    *i;
    size_t ki = i.ki;
    size_t vi = i.vi;
    --msize;
    if (vi == -1) {
      Key* keylist = keys2[ki];
      size_t s = sizes2[ki];
      if (s) {
        --s;
        sizes2[ki] = s;
        Key* k = &keylist[s];
        Value* v = &values2[ki][s];
        keys[ki] = std::move(*k);
        values[ki] = std::move(*v);
        k->~Key();
        v->~Value();
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
      size_t s = sizes2[ki] - 1;
      Key* keylist = keys2[ki];
      Value* valuelist = values2[ki];
      if (s == vi) {
        ++i;
      } else {
        keylist[vi] = std::move(keylist[s]);
        valuelist[vi] = std::move(valuelist[s]);
      }
      keylist[s].~Key();
      valuelist[s].~Value();
      sizes2[ki] = s;
      if (i != end()) {
        *i;
      }
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
      int hits = 0;
      for (size_t i = 0; i != bs; ++i) {
        if (oldHasKey[i]) {
          insert(std::move(oldKeys[i]), std::move(oldValues[i]));
          --msize;
          oldKeys[i].~Key();
          oldValues[i].~Value();
          ++hits;
        }
      }
      //printf("hits: %d/%d  misses: %d\n", hits, bs, bs - hits);
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
    Key** oldKeys2 = keys2;
    Value** oldValues2 = values2;
    size_t* oldSizes2 = sizes2;
    size_t* oldAllocated2 = allocated2;

    keys2 = allocate<Key*>(newBs);
    values2 = allocate<Value*>(newBs);
    sizes2 = allocate<size_t>(newBs);
    allocated2 = allocate<size_t>(newBs);

    std::memset(sizes2, 0, sizeof(size_t) * newBs);
    std::memset(keys2, 0, sizeof(Key*) * newBs);

    if (oldKeys2) {
      size_t bs = ksize;
      ksize = newBs;
      for (size_t ki = 0; ki != bs; ++ki) {
        Key* keylist = oldKeys2[ki];
        if (keylist) {
          Value* valuelist = oldValues2[ki];
          size_t s = oldSizes2[ki];
          for (size_t vi = 0; vi != s; ++vi) {
            insert(std::move(keylist[vi]), std::move(valuelist[vi]));
            --msize;
            oldKeys2[ki][vi].~Key();
            oldValues2[ki][vi].~Value();
          }
          deallocate(keylist);
          deallocate(valuelist);
        }
      }
      ksize = bs;
    }

    deallocate(oldKeys2);
    deallocate(oldValues2);
    deallocate(oldSizes2);
    deallocate(oldAllocated2);
  }

  template<typename KeyT>
  iterator find(KeyT&& key) noexcept {
    size_t bs = ksize;
    if (bs == 0) {
      return end();
    }
    size_t ki = Hash()(key) & (bs - 1);
    if (hasKey && hasKey[ki] && keys[ki] == key) {
      return iterator(this, ki, -1, &values[ki]);
    }
    Key* keylist = keys2[ki];
    if (keylist) {
      size_t s = sizes2[ki];
      for (size_t vi = 0; vi != s; ++vi) {
        if (keylist[vi] == key) {
          //fmt::printf("map %p found key %#x\n", (void*)this, key);
          return iterator(this, ki, vi, &values2[ki][vi]);
        }
      }
    }
    //fmt::printf("map %p no find key %#x\n", (void*)this, key);
    return end();
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
    size_t bs = ksize;
    size_t ki = Hash()(key) & (bs - 1);
    while (true) {
      if (hasKey[ki]) {
        if (keys[ki] == key) {
          return std::make_pair(iterator(this, ki, -1, &values[ki]), false);
        }
      } else {
        ++msize;
        new (&keys[ki]) Key(std::forward<KeyT>(key));
        new (&values[ki]) Value(std::forward<Args>(args)...);
        //printf("constructed at %p\n", (void*)&values[ki]);
        hasKey[ki] = true;
        return std::make_pair(iterator(this, ki, -1, &values[ki]), true);
      }
      if (bs < msize) {
        bs *= 2;
        rehash1(bs);
        rehash2(bs);
        ksize = bs;
        ki = Hash()(key) & (bs - 1);
        continue;
      }
      break;
    }
    size_t s;
    //printf("keys2 is %p\n", (void*)keys2);
    Key* ok = keys2[ki];
    //printf("ok is %p\n", (void*)ok);
    Value* ov = values2[ki];
    if (!ok) {
      ok = allocate<Key>(1);
      ov = allocate<Value>(1);
      //printf("allocate for %d\n", ki);
      keys2[ki] = ok;
      values2[ki] = ov;
      sizes2[ki] = 0;
      allocated2[ki] = 1;
      s = 0;
    } else {
      s = sizes2[ki];
      for (size_t vi = 0; vi != s; ++vi) {
        if (ok[vi] == key) {
          return std::make_pair(iterator(this, ki, vi, &values2[ki][vi]), false);
        }
      }
      size_t a = allocated2[ki];
      if (s == a) {
        size_t na = a + (a + 1) / 2;
        Key* nk = allocate<Key>(na);
        Value* nv = allocate<Value>(na);
        for (size_t vi = 0; vi != s; ++vi) {
          new (&nk[vi]) Key(std::move(ok[vi]));
          new (&nv[vi]) Value(std::move(ov[vi]));
          ok[vi].~Key();
          ov[vi].~Value();
        }
        deallocate(ok);
        deallocate(ov);
        keys2[ki] = nk;
        values2[ki] = nv;
        allocated2[ki] = na;
        ok = nk;
        ov = nv;
      }
    }
    ++msize;
    //fmt::printf("ctor?\n");
    new (&ok[s]) Key(std::forward<KeyT>(key));
    new (&ov[s]) Value(std::forward<Args>(args)...);
    //fmt::printf("ctor done!\n");
    sizes2[ki] = s + 1;
    return std::make_pair(iterator(this, ki, s, &ov[s]), true);
  }

  size_t bucket_count() const noexcept {
    return ksize;
  }
  size_t size() const noexcept {
    return msize;
  }

};

#endif

}