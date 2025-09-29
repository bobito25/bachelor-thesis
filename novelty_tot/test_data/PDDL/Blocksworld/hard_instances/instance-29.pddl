

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b a)
(ontable c)
(on d e)
(on e c)
(on f b)
(clear f)
)
(:goal
(and
(on b a)
(on c e)
(on d f)
(on e b))
)
)


